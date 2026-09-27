/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "udf_twl_timer.h"
#include "udf_twl_timer_inner.h"

#include <inttypes.h>

#include "securec.h"
#include "udf_timer_adapt.h"

#ifdef __cplusplus
extern "C" {
#endif

#define UDF_TML_TIMER_PROC_NAME_LEN 32
#define UDF_TWLTMR_INTERVAL_INVALID 0xFFFFFFFF

#define UDF_TWL_TIMER_MAGIC 0x55574C30
#define UDF_TWL_TIMER_INST_MAGIC 0x55574C31
#define UDF_TWL_TIMER_WHEEL_INST_MAGIC 0x55574C32
#define UDF_TWL_TIMER_ATTR_MAGIC 0x55574C33
#define UDF_TWL_TIMER_DEFAULT_NAME "udfTwlTimer"
#define UDF_TWL_TIMER_INST_DEFAULT_NAME "udfTwlTimerInst"
#define UDF_TWL_TIMER_INST_TREATE_NAME "udfTwlInstDeal"

#define UDF_TWL_TIMER_MSEC_INTERVAL 100
#define UDF_TWL_TIMER_SEC_INTERVAL 1000
#define UDF_TWL_TIMER_MINUTE_INTERVAL (60 * 1000)
#define UDF_TWL_TIMER_HOUR_INTERVAL (60 * 60 * 1000)

#define UDF_TWL_TIMERMSEC_SLOT_NUM 10  /* time wheel slot num  millisecond timer */
#define UDF_TWL_TIMERSEC_SLOT_NUM 60   /* time wheel slot num for second timer */
#define UDF_TWL_TIMERMIN_SLOT_NUM 60   /* time wheel slot num  for min timer */
#define UDF_TWL_TIMERHOUR_SLOT_NUM 200 /* time wheel slot num  for hour timer */

#define UDF_TWL_PRECISE_NUM 4
#define UDF_TWL_PRECISE_MSEC (0x01 << 0)
#define UDF_TWL_PRECISE_SEC (0x01 << 1)
#define UDF_TWL_PRECISE_MINUTE (0x01 << 2)
#define UDF_TWL_PRECISE_HOUR (0x01 << 3)
#define UDF_TWL_PRECISE_FACTOR5 (5)
#define UDF_TWL_OUTSIZE_40 (40)
#define UDF_TWL_OUTSIZE_120 (120)
#define UDF_TWL_NUM2 (2)
#pragma pack(4)

/**
 * @brief 时间轮层级状态
 */
typedef enum {
    UDF_TWL_TIMER_WHEEL_MSEC,   /* 毫秒级时间轮 */
    UDF_TWL_TIMER_WHEEL_SEC,    /* 秒级时间轮 */
    UDF_TWL_TIMER_WHEEL_MINUTE, /* 分钟级时间轮 */
    UDF_TWL_TIMER_WHEEL_HOUR,   /* 小时级时间轮 */
    UDF_TWL_TIMER_WHEEL_BUTT
} UdfTwlTimerWheelType;

/**
 * @brief 定时器实例状态
 */
typedef enum {
    UDF_TWL_TIMER_INST_WAIT_EXECUTE, /* 定时器实例等待执行 */
    UDF_TWL_TIMER_INST_LOAD,         /* 定时器实例加载 */
    UDF_TWL_TIMER_INST_BUTT
} UdfTwlTimerInstAddStage;

typedef struct {
    uint32_t magic;
    uint32_t slot_num;         /* 该层时间轮格子总数 */
    uint32_t cur_slot;         /* 该层时间轮运行指向的格子位置 */
    VOS_LIST_HEAD_S *listHead; /* 该层时间轮链表头 */
    uint32_t timerCount;       /* 该层时间轮挂载的定时器数量 */
    UdfTwlTimerWheelType type; /* 该层时间轮类型 */
} UdfTwlTimerWheel;

/* Timer wheel instance */
typedef struct {
    uint32_t magic;                                        /* 时间轮定时器魔术字 */
    uint32_t creatCnt;                                     /* 时间轮创建次数 */
    BaseTimerHandle timerHd;                               /* 时间轮定时器句柄 */
    UdfTwlTimerWheel *wheelList[UDF_TWL_TIMER_WHEEL_BUTT]; /* 各级时间轮 */
    uint8_t precision;                                     /* 时间轮定时器实例精度 */
    UdfTwlTimerWheelType execLevel;                        /* 时间轮定时器执行层级 */
    VOS_LIST_HEAD_S timerlist;                             /* 定时器链表(所有的定时器节点) */
    char name[UDF_TWL_TMR_NAME_LEN];                       /* 定时器名称 */
    TimerAppInfo appInfo;
} UdfTwlTimerWheelCtrl;

typedef struct {
    uint64_t timeOutCount;
    uint64_t lastSysTime;
    uint32_t maxProcTime;
    uint32_t maxInterval;
    uint32_t minInterval;
    char lastTime[BASE_TIMER_LAST_PROC_TIME_LEN];
} UdfTwlTimerInstDbgInfo;

/**
 * @brief 时间轮实例运行状态
 */
typedef enum {
    UDF_TWLTMR_INST_STATUS_RUN,  /* 运行状态 */
    UDF_TWLTMR_INST_STATUS_DEL,  /* 销毁状态 */
    UDF_TWLTMR_INST_STATUS_STOP, /* 暂停状态 */
    UDF_TWLTMR_INST_STATUS_BUTT
} UdfTmrInstStatus;

/* Timer node */
typedef struct {
    uint32_t magic;
    VOS_LIST_NODE_S node;                 /* 定时器节点(所属层级时间轮链表) */
    VOS_LIST_NODE_S nodeForMT;            /* 定时器节点(定时器链表) */
    char timername[UDF_TWL_TMR_NAME_LEN]; /* 定时器实例名称 */
    UdfTmrInstStatus status;              /* 定时器实例运行状态 */
    bool begingSchedule;                  /* 定时器是否正在执行超时回调 */
    uint32_t resizeInterval;              /* 定时器实例重置超时时间 */
    uint32_t realInterval;                /* 定时器实例超时时间 */
    uint32_t interval;                    /* 定时器实例规范化超时时间 */
    uint32_t rotation;                    /* 定时器实例运行圈数 */
    UdfTmrInstCallBack proc;              /* 定时器实例回调函数 */
    void *usrData;                        /* 定时器实例用户自定义数据 */
    UdfTimerMode tmrMode;                 /* 定时器实例周期性：单次或周期 */
    uint32_t index;                       /* 定时器实例在时间轮上的位置 */
    UdfTwlTimerWheelType precision;       /* 定时器实例精度 */
    UdfTwlTimerWheel *tmrWheel;           /* 定时器实例部署的时间轮层级 */
    UdfTwlTimerWheelCtrl *WheelHandle;    /* 定时器实例部署的时间轮 */
    UdfTwlTimerInstDbgInfo dbgInfo;       /* 定时器实例维测信息 */
    TimerDestructProc destructProc;       /* 定时器销毁时析构回调 */
} UdfTwlTimerInst;

typedef struct {
    uint32_t magic;
    char TwlTimerName[UDF_TWL_TMR_NAME_LEN];
} UdfTwlTimerParam;

typedef struct {
    UdfTwlTimerWheelType level; /* 时间轮级别 */
    uint8_t precision;          /* 定时器实例精度 */
    uint32_t slotNum;           /* 时间轮对应的格子数量 */
    uint32_t interval;          /* 时间轮对应的时间间隔 */
} UdfTwlTimerInitTransferMap;

typedef uint32_t (*UdfTwlTimerSetAttrProc)(UdfTwlTimerParam *attr, const void *val, size_t len);
uint32_t UdfTwlTimerInitWheel(UdfTwlTimerWheelCtrl *wheelCtrl, UdfTwlTimerWheelType type);

typedef struct {
    VOS_LIST_HEAD_S instList; /* 定时器并发修改保护 */
    VOS_LIST_HEAD_S tmrList;  /* 定时器并发修改保护 */
} UdfTwlTimerCtrl;

#pragma pack()

void UdfTimerInstMoveNode(UdfTwlTimerWheel *timerWheel, UdfTwlTimerInst *tmrInst)
{
    if (timerWheel == NULL) {
        return;
    }

    VOS_ListRemove(&tmrInst->node);
    if (timerWheel->timerCount > 0) {
        timerWheel->timerCount--;
    }
    return;
}

void UdfTimerInstDelNode(UdfTwlTimerWheelCtrl *wheelCtrl, UdfTwlTimerWheel *timerWheel, UdfTwlTimerInst *tmrInst)
{
    TimerDestructProc destructProc = NULL;
    void *usrParam = NULL;

    if ((tmrInst == NULL) || (tmrInst->magic != UDF_TWL_TIMER_INST_MAGIC)) {
        return;
    }

    if ((tmrInst->usrData != NULL) && (tmrInst->destructProc != NULL)) {
        destructProc = tmrInst->destructProc;
        usrParam = tmrInst->usrData;
    }

    UdfTimerInstMoveNode(timerWheel, tmrInst);
    VOS_ListRemove(&tmrInst->nodeForMT);
    tmrInst->magic = 0;
    Timer_MemFree(&wheelCtrl->appInfo, tmrInst);

    if (destructProc != NULL) {
        destructProc(usrParam);
    }

    return;
}

UdfTwlTimerWheel *UdfTwlTimerGetRunningWheel(UdfTwlTimerWheelCtrl *wheelCtrl, UdfTwlTimerWheelType level)
{
    UdfTwlTimerWheel *timerWheel = NULL;

    if (level >= UDF_TWL_TIMER_WHEEL_BUTT) {
        return NULL;
    }

    timerWheel = wheelCtrl->wheelList[level];

    return timerWheel;
}

void UdfTwlTimerDestoryWheelInst(UdfTwlTimerWheelCtrl *wheelCtrl, UdfTwlTimerWheel *timerWheel)
{
    UdfTwlTimerInst *timerInst = NULL;
    uint32_t uiIndex;
    VOS_LIST_HEAD_S *dllNode = NULL;
    VOS_LIST_HEAD_S *nextDllNode = NULL;

    if (timerWheel == NULL) {
        return;
    }

    for (uiIndex = 0; uiIndex < timerWheel->slot_num; uiIndex++) {
        VOS_LIST_FOR_EACH_ITEM_SAFE(dllNode, nextDllNode, &(timerWheel->listHead[uiIndex]))
        {
            timerInst = VOS_LIST_ENTRY(dllNode, UdfTwlTimerInst, node);
            if (timerInst == NULL) {
                continue;
            }

            UdfTimerInstDelNode(wheelCtrl, timerWheel, timerInst);
        }
    }
    Timer_MemFree(&wheelCtrl->appInfo, timerWheel->listHead);
    Timer_MemFree(&wheelCtrl->appInfo, timerWheel);
}

void UdfTwlTimerDestoryWheel(void *param)
{
    UdfTwlTimerWheelCtrl *wheelCtrl = (UdfTwlTimerWheelCtrl *)param;
    if (wheelCtrl == NULL || wheelCtrl->magic != UDF_TWL_TIMER_MAGIC) {
        return;
    }

    for (uint32_t i = 0; i < (uint32_t)UDF_TWL_TIMER_WHEEL_BUTT; i++) {
        UdfTwlTimerDestoryWheelInst(wheelCtrl, wheelCtrl->wheelList[i]);
    }

    Timer_MemFree(&wheelCtrl->appInfo, wheelCtrl);

    return;
}

uint32_t UdfTwlTimerGetIndex(UdfTwlTimerWheelType type)
{
    uint32_t i;
    UdfTwlTimerWheelType wheelList[] = {UDF_TWL_TIMER_WHEEL_MSEC, UDF_TWL_TIMER_WHEEL_SEC, UDF_TWL_TIMER_WHEEL_MINUTE,
        UDF_TWL_TIMER_WHEEL_HOUR, UDF_TWL_TIMER_WHEEL_BUTT};

    if (type >= UDF_TWL_TIMER_WHEEL_BUTT) {
        return UDF_ERROR;
    }

    for (i = 0; i < sizeof(wheelList) / sizeof(UdfTwlTimerWheelType); i++) {
        if (type == wheelList[i]) {
            return i;
        }
    }

    return UDF_ERROR;
}

uint32_t UdfTwlTimerCorrectWheel(UdfTwlTimerWheelCtrl *wheelCtrl, uint32_t remainTime, UdfTwlTimerWheelType level)
{
    int32_t i;
    uint32_t slotNum;
    uint32_t timeTmp = remainTime;
    int32_t typeStart = (int32_t)UdfTwlTimerGetIndex(wheelCtrl->execLevel);
    int32_t typeEnd = (int32_t)UdfTwlTimerGetIndex(level);
    UdfTwlTimerWheel *timerWheel = NULL;
    UdfTwlTimerInitTransferMap wheelList[] = {
        {UDF_TWL_TIMER_WHEEL_MSEC, UDF_TWL_PRECISE_MSEC, UDF_TWL_TIMERMSEC_SLOT_NUM, UDF_TWL_TIMER_MSEC_INTERVAL},
        {UDF_TWL_TIMER_WHEEL_SEC, UDF_TWL_PRECISE_SEC, UDF_TWL_TIMERSEC_SLOT_NUM, UDF_TWL_TIMER_SEC_INTERVAL},
        {UDF_TWL_TIMER_WHEEL_MINUTE, UDF_TWL_PRECISE_MINUTE, UDF_TWL_TIMERMIN_SLOT_NUM, UDF_TWL_TIMER_MINUTE_INTERVAL},
        {UDF_TWL_TIMER_WHEEL_HOUR, UDF_TWL_PRECISE_HOUR, UDF_TWL_TIMERHOUR_SLOT_NUM, UDF_TWL_TIMER_HOUR_INTERVAL},
    };

    if (wheelCtrl->execLevel == UDF_TWL_TIMER_WHEEL_BUTT) {
        return UDF_OK;
    }

    if ((typeStart == (int32_t)UDF_ERROR) || (typeEnd == (int32_t)UDF_ERROR)) {
        return UDF_ERROR;
    }

    for (i = typeStart - 1; i >= typeEnd; i--) {
        timerWheel = UdfTwlTimerGetRunningWheel(wheelCtrl, wheelList[i].level);
        if (timerWheel == NULL) {
            LOG_INNER_ERR("UDF init timer wheel correct remain time unsuccessfully.");
            return UDF_ERROR;
        }

        slotNum = timeTmp / wheelList[i].interval;
        timeTmp -= slotNum * wheelList[i].interval;
        timerWheel->cur_slot = timerWheel->slot_num - slotNum;
    }

    return UDF_OK;
}

void UdfTimerInstGetExectueWheel(UdfTwlTimerWheelCtrl *wheelCtrl, UdfTwlTimerInst *timerInst, uint32_t *instIndex,
    uint32_t *typeEnd)
{
    (void)wheelCtrl;
    /* 因为只有获取remainTime的时候，不在业务指定线程上运行。为了节约内存空间，这里使用时间轮wheelCtrl->listLock的锁加强防护
     */
    *instIndex = timerInst->index;
    *typeEnd = UdfTwlTimerGetIndex(timerInst->tmrWheel->type);
}

uint32_t UdfTwlTimerCalcRemainTime(UdfTwlTimerWheelCtrl *wheelCtrl, UdfTwlTimerInst *timerInst, uint32_t *remainTime)
{
    uint32_t i;
    uint32_t typeEnd;
    uint32_t curSlot;
    uint32_t instIndex;
    uint32_t timeTmp = 0;
    UdfTwlTimerWheel *timerWheel = NULL;
    if (wheelCtrl == NULL) {
        return UDF_ERROR;
    }
    uint32_t typeBegin = UdfTwlTimerGetIndex(wheelCtrl->execLevel);
    UdfTimerInstGetExectueWheel(wheelCtrl, timerInst, &instIndex, &typeEnd);
    UdfTwlTimerInitTransferMap wheelList[] = {
        {UDF_TWL_TIMER_WHEEL_MSEC, UDF_TWL_PRECISE_MSEC, UDF_TWL_TIMERMSEC_SLOT_NUM, UDF_TWL_TIMER_MSEC_INTERVAL},
        {UDF_TWL_TIMER_WHEEL_SEC, UDF_TWL_PRECISE_SEC, UDF_TWL_TIMERSEC_SLOT_NUM, UDF_TWL_TIMER_SEC_INTERVAL},
        {UDF_TWL_TIMER_WHEEL_MINUTE, UDF_TWL_PRECISE_MINUTE, UDF_TWL_TIMERMIN_SLOT_NUM, UDF_TWL_TIMER_MINUTE_INTERVAL},
        {UDF_TWL_TIMER_WHEEL_HOUR, UDF_TWL_PRECISE_HOUR, UDF_TWL_TIMERHOUR_SLOT_NUM, UDF_TWL_TIMER_HOUR_INTERVAL},
    };

    if (wheelCtrl->execLevel == UDF_TWL_TIMER_WHEEL_BUTT) {
        return UDF_OK;
    }

    bool typeInvalid = ((typeBegin == (uint32_t)UDF_ERROR) || (typeEnd == (uint32_t)UDF_ERROR) ||
                        (typeEnd > UDF_TWL_TIMER_WHEEL_HOUR));
    if (typeInvalid) {
        return UDF_ERROR;
    }

    for (i = typeBegin; i <= typeEnd; i++) {
        timerWheel = UdfTwlTimerGetRunningWheel(wheelCtrl, wheelList[i].level);
        if (timerWheel == NULL) {
            LOG_INNER_ERR("UDF init timer wheel correct remain time unsuccessfully.");
            return UDF_ERROR;
        }

        curSlot = timerWheel->cur_slot;
        if (i < typeEnd) {
            timeTmp += (timerWheel->slot_num - curSlot) * wheelList[i].interval;
            continue;
        }

        if (curSlot == instIndex) {
            continue;
        } else if (curSlot < instIndex) {
            timeTmp += (instIndex - curSlot - 1) * wheelList[i].interval;
        } else {
            timeTmp += (instIndex + timerWheel->slot_num - curSlot - 1) * wheelList[i].interval;
        }
    }

    *remainTime = timeTmp;
    return UDF_OK;
}

uint32_t UdfTwlTimerResizeBaseTimer(UdfTwlTimerWheelCtrl *wheelCtrl, uint32_t interval, UdfTwlTimerWheelType level)
{
    uint32_t ret;
    uint32_t remainTime;

    if (wheelCtrl->timerHd == NULL) {
        return UDF_OK;
    }

    if (wheelCtrl->execLevel <= level) {
        return UDF_OK;
    }

    ret = BASE_TimerGetRemainTime(wheelCtrl->timerHd, &remainTime);
    if (ret != UDF_OK) {
        LOG_INNER_ERR("UDF init timer wheel get remain time unsuccessfully.");
        return UDF_ERROR;
    }

    ret = UdfTwlTimerCorrectWheel(wheelCtrl, remainTime, level);
    if (ret != UDF_OK) {
        LOG_INNER_ERR("UDF init timer wheel correct remain time unsuccessfully.");
        return UDF_ERROR;
    }

    ret = BASE_TimerResize(wheelCtrl->timerHd, interval);
    if (ret != UDF_OK) {
        LOG_INNER_ERR("UDF init timer wheel reset timeout interval unsuccessfully.");
        return UDF_ERROR;
    }

    return UDF_OK;
}

uint32_t UdfTwlTimerInitWheelLost(UdfTwlTimerWheelCtrl *wheelCtrl, uint32_t typeBegin, uint32_t typeEnd)
{
    uint32_t i;
    UdfTwlTimerWheel *timerWheel = NULL;
    UdfTwlTimerWheelType wheelList[] = {
        UDF_TWL_TIMER_WHEEL_MSEC, UDF_TWL_TIMER_WHEEL_SEC, UDF_TWL_TIMER_WHEEL_MINUTE, UDF_TWL_TIMER_WHEEL_HOUR};

    for (i = typeBegin + 1; i < typeEnd; i++) {
        timerWheel = UdfTwlTimerGetRunningWheel(wheelCtrl, wheelList[i]);
        if (timerWheel != NULL) {
            continue;
        }

        if (UdfTwlTimerInitWheel(wheelCtrl, wheelList[i]) != UDF_OK) {
            return UDF_ERROR;
        }
    }

    return UDF_OK;
}

uint32_t UdfTwlTimerCheckWheelValid(UdfTwlTimerWheelCtrl *wheelCtrl, UdfTwlTimerWheelType type)
{
    (void)type;
    uint32_t i;
    uint32_t ret;
    uint32_t lowLevelWheel = (uint32_t)-1;
    UdfTwlTimerWheel *timerWheel = NULL;
    UdfTwlTimerWheelType wheelList[] = {
        UDF_TWL_TIMER_WHEEL_MSEC, UDF_TWL_TIMER_WHEEL_SEC, UDF_TWL_TIMER_WHEEL_MINUTE, UDF_TWL_TIMER_WHEEL_HOUR};

    for (i = 0; i < sizeof(wheelList) / sizeof(UdfTwlTimerWheelType); i++) {
        timerWheel = UdfTwlTimerGetRunningWheel(wheelCtrl, wheelList[i]);
        if (timerWheel == NULL) {
            continue;
        }

        if ((lowLevelWheel != (uint32_t)-1) && (i - lowLevelWheel) >= UDF_TWL_NUM2) {
            ret = UdfTwlTimerInitWheelLost(wheelCtrl, lowLevelWheel, i);
            if (ret != UDF_OK) {
                return UDF_ERROR;
            }
        }

        lowLevelWheel = i;
    }

    return UDF_OK;
}

UdfTwlTimerWheel *UdfTwlTimerInitWheelInst(UdfTwlTimerWheelCtrl *wheelCtrl, uint32_t interval, uint32_t num,
    UdfTwlTimerWheelType level)
{
    (void)interval;
    uint32_t ret;
    uint32_t i = 0;
    UdfTwlTimerWheel *timerWheel = NULL;

    if ((num == 0) || (level == UDF_TWL_TIMER_WHEEL_BUTT)) {
        return NULL;
    }

    timerWheel = Timer_MemAlloc(&wheelCtrl->appInfo, sizeof(UdfTwlTimerWheel));
    if (timerWheel == NULL) {
        return NULL;
    }

    (void)memset_s(timerWheel, sizeof(UdfTwlTimerWheel), 0, sizeof(UdfTwlTimerWheel));

    timerWheel->listHead = (VOS_LIST_HEAD_S *)Timer_MemAlloc(&wheelCtrl->appInfo, num * sizeof(VOS_LIST_HEAD_S));
    if (timerWheel->listHead == NULL) {
        Timer_MemFree(&wheelCtrl->appInfo, timerWheel);
        return NULL;
    }

    while (i < num) {
        VOS_ListInit(&(timerWheel->listHead[i]));
        i++;
    }

    timerWheel->slot_num = num;
    timerWheel->magic = UDF_TWL_TIMER_WHEEL_INST_MAGIC;
    timerWheel->type = level;

    wheelCtrl->wheelList[level] = timerWheel;
    ret = UdfTwlTimerCheckWheelValid(wheelCtrl, level);
    if (ret != UDF_OK) {
        LOG_INNER_ERR("UDF init timer wheel, wheel create invalid.");
        UdfTwlTimerDestoryWheelInst(wheelCtrl, timerWheel);
        wheelCtrl->wheelList[level] = NULL;
        return NULL;
    }

    return timerWheel;
}

uint32_t UdfTwlTimerInitWheel(UdfTwlTimerWheelCtrl *wheelCtrl, UdfTwlTimerWheelType type)
{
    uint32_t i;
    uint32_t ret;
    uint32_t slotNum = 0;
    uint32_t interval = 0;
    uint8_t precision = 0;
    UdfTwlTimerWheel *timerWheel = NULL;

    UdfTwlTimerInitTransferMap transferMap[] = {
        {UDF_TWL_TIMER_WHEEL_MSEC, UDF_TWL_PRECISE_MSEC, UDF_TWL_TIMERMSEC_SLOT_NUM, UDF_TWL_TIMER_MSEC_INTERVAL},
        {UDF_TWL_TIMER_WHEEL_SEC, UDF_TWL_PRECISE_SEC, UDF_TWL_TIMERSEC_SLOT_NUM, UDF_TWL_TIMER_SEC_INTERVAL},
        {UDF_TWL_TIMER_WHEEL_MINUTE, UDF_TWL_PRECISE_MINUTE, UDF_TWL_TIMERMIN_SLOT_NUM, UDF_TWL_TIMER_MINUTE_INTERVAL},
        {UDF_TWL_TIMER_WHEEL_HOUR, UDF_TWL_PRECISE_HOUR, UDF_TWL_TIMERHOUR_SLOT_NUM, UDF_TWL_TIMER_HOUR_INTERVAL},
    };

    if (type >= UDF_TWL_TIMER_WHEEL_BUTT) {
        return UDF_ERROR;
    }

    for (i = 0; i < sizeof(transferMap) / sizeof(UdfTwlTimerInitTransferMap); i++) {
        if (transferMap[i].level == type) {
            slotNum = transferMap[i].slotNum;
            precision = transferMap[i].precision;
            interval = transferMap[i].interval;
            break;
        }
    }

    timerWheel = UdfTwlTimerInitWheelInst(wheelCtrl, interval, slotNum, type);
    if (timerWheel == NULL) {
        LOG_INNER_ERR("UDF init timer wheel inst unsuccessfully.");
        return UDF_ERROR;
    }

    ret = UdfTwlTimerResizeBaseTimer(wheelCtrl, interval, type);
    if (ret != UDF_OK) {
        LOG_INNER_ERR("UDF init timer wheel reset timeout interval unsuccessfully.");
        UdfTwlTimerDestoryWheelInst(wheelCtrl, timerWheel);
        wheelCtrl->wheelList[type] = NULL;
        return UDF_ERROR;
    }

    wheelCtrl->precision |= precision;
    for (i = 0; i < sizeof(transferMap) / sizeof(UdfTwlTimerInitTransferMap); i++) {
        if ((wheelCtrl->precision & transferMap[i].precision) == transferMap[i].precision) {
            wheelCtrl->execLevel = transferMap[i].level;
            break;
        }
    }

    return UDF_OK;
}

uint32_t UdfTimerInstAddNode(UdfTwlTimerWheelCtrl *wheelCtrl, UdfTwlTimerWheelType level, UdfTwlTimerInst *timerInst,
    uint32_t adjustNum, UdfTwlTimerInstAddStage stage)
{
    uint32_t ret;
    uint32_t slotIndex;
    uint64_t startTime = 0;
    uint32_t rotation = 0;
    UdfTwlTimerWheel *timerWheel = NULL;
    uint32_t adjustNumReal;
    uint32_t curSlot;

    timerWheel = UdfTwlTimerGetRunningWheel(wheelCtrl, level);
    if (timerWheel == NULL) {
        ret = UdfTwlTimerInitWheel(wheelCtrl, level);
        if (ret != UDF_OK) {
            LOG_INNER_ERR("UDF init timer wheel unsuccessfully.");
            return UDF_ERROR;
        }

        timerWheel = UdfTwlTimerGetRunningWheel(wheelCtrl, level);
        if (timerWheel == NULL) {
            return UDF_ERROR;
        }
    }

    if (timerWheel->slot_num == 0) {
        LOG_INNER_ERR("slot_num:%u not correct.", timerWheel->slot_num);
        return UDF_ERROR;
    }

    timerInst->WheelHandle = wheelCtrl;
    adjustNumReal = (adjustNum > timerWheel->slot_num) ? timerWheel->slot_num : adjustNum;

    curSlot = timerWheel->cur_slot;
    if (stage == UDF_TWL_TIMER_INST_WAIT_EXECUTE) {
        /* 根据定时器实例调整参数计算定时器实例插入的格子位置 */
        slotIndex = (curSlot + adjustNumReal) % timerWheel->slot_num;
    } else {
        /* 根据定时器实例超时时间和调整参数计算定时器实例插入的格子位置及圈数 */
        rotation = (timerInst->interval + adjustNumReal) / timerWheel->slot_num;
        slotIndex = (curSlot + timerInst->interval + adjustNumReal) % timerWheel->slot_num;
    }

    VOS_ListInit(&timerInst->node);
    VOS_ListAdd(&(timerInst->node), &(timerWheel->listHead[slotIndex]));
    timerWheel->timerCount++;
    timerInst->rotation = rotation;
    timerInst->tmrWheel = timerWheel;
    timerInst->index = slotIndex;
    if (timerInst->dbgInfo.lastSysTime == 0) {
        (void)SystimeGetMilliSec(&startTime);
        timerInst->dbgInfo.lastSysTime = startTime;
    }

    return UDF_OK;
}

uint32_t UdfTimerInstCalcPrecision(UdfTwlTimerWheelCtrl *wheelCtrl, UdfTwlTimerInst *timerInst, uint32_t interval)
{
    uint32_t i;
    uint32_t ret;
    UdfTwlTimerWheelType level = UDF_TWL_TIMER_WHEEL_HOUR;
    uint32_t basicInterval = UDF_TWL_TIMER_HOUR_INTERVAL;

    UdfTwlTimerInitTransferMap transferMap[] = {
        {UDF_TWL_TIMER_WHEEL_MSEC, UDF_TWL_PRECISE_MSEC, UDF_TWL_TIMERMSEC_SLOT_NUM, UDF_TWL_TIMER_MSEC_INTERVAL},
        {UDF_TWL_TIMER_WHEEL_SEC, UDF_TWL_PRECISE_SEC, UDF_TWL_TIMERSEC_SLOT_NUM, UDF_TWL_TIMER_SEC_INTERVAL},
        {UDF_TWL_TIMER_WHEEL_MINUTE, UDF_TWL_PRECISE_MINUTE, UDF_TWL_TIMERMIN_SLOT_NUM, UDF_TWL_TIMER_MINUTE_INTERVAL},
        {UDF_TWL_TIMER_WHEEL_HOUR, UDF_TWL_PRECISE_HOUR, UDF_TWL_TIMERHOUR_SLOT_NUM, UDF_TWL_TIMER_HOUR_INTERVAL},
    };

    if (timerInst->status == UDF_TWLTMR_INST_STATUS_STOP) {
        return UDF_OK;
    }

    for (i = 0; i < sizeof(transferMap) / sizeof(UdfTwlTimerInitTransferMap); i++) {
        if (interval < transferMap[i].slotNum * transferMap[i].interval) {
            level = transferMap[i].level;
            basicInterval = transferMap[i].interval;
            break;
        }
    }

    timerInst->precision = level;
    timerInst->interval = interval / basicInterval;
    timerInst->realInterval = interval;
    timerInst->resizeInterval = UDF_TWLTMR_INTERVAL_INVALID;
    ret = UdfTimerInstAddNode(wheelCtrl, level, timerInst, 0, UDF_TWL_TIMER_INST_LOAD);
    if (ret != UDF_OK) {
        LOG_INNER_ERR("Twl timer inst add in twl timer list failed.");
        return UDF_ERROR;
    }

    return UDF_OK;
}

void UdfTwlTimerReloadInst(UdfTwlTimerWheelCtrl *wheelCtrl, UdfTwlTimerInst *timerInst, UdfTwlTimerWheel *timerWheel)
{
    uint32_t ret;
    if ((timerInst->tmrMode == UDF_TIMER_MODE_PERIOD) && (timerInst->status != UDF_TWLTMR_INST_STATUS_DEL)) {
        UdfTimerInstMoveNode(timerWheel, timerInst);
        if ((timerInst->status == UDF_TWLTMR_INST_STATUS_RUN) &&
            (timerInst->resizeInterval == UDF_TWLTMR_INTERVAL_INVALID)) {
            (void)UdfTimerInstAddNode(wheelCtrl, timerInst->precision, timerInst, 0, UDF_TWL_TIMER_INST_LOAD);
        } else {
            VOS_ListRemove(&timerInst->nodeForMT);
            VOS_ListAdd(&(timerInst->nodeForMT), &(wheelCtrl->timerlist));
            ret = UdfTimerInstCalcPrecision(wheelCtrl, timerInst, timerInst->resizeInterval);
            if (ret != UDF_OK) {
                LOG_INNER_ERR("Udf timer instance timeout interval resize unsuccess. (0x%x).", ret);
            }
        }
        return;
    }

    UdfTimerInstDelNode(wheelCtrl, timerWheel, timerInst);
    return;
}

void UdfTwlTimerUpdateDbgInfo(UdfTwlTimerInst *timerInst, uint64_t startTime, uint64_t endTime, char *timeOutTime)
{
    uint32_t interval = 0;
    uint32_t procTime = 0;

    if (endTime < startTime) {
        return;
    }

    procTime = endTime - startTime;
    if (procTime > timerInst->dbgInfo.maxProcTime) {
        timerInst->dbgInfo.maxProcTime = procTime;
    }

    if (timerInst->dbgInfo.lastSysTime == 0) {
        timerInst->dbgInfo.lastSysTime = startTime;
    } else {
        interval = startTime - timerInst->dbgInfo.lastSysTime;
        if (interval > timerInst->dbgInfo.maxInterval) {
            timerInst->dbgInfo.maxInterval = interval;
        }

        if ((timerInst->dbgInfo.minInterval == 0) || (interval < timerInst->dbgInfo.minInterval)) {
            timerInst->dbgInfo.minInterval = interval;
        }

        timerInst->dbgInfo.lastSysTime = startTime;
    }

    errno_t errNo = strcpy_s(timerInst->dbgInfo.lastTime, BASE_TIMER_LAST_PROC_TIME_LEN, timeOutTime);
    if (errNo != EOK) {
        return;
    }
    timerInst->dbgInfo.timeOutCount++;

    return;
}

typedef struct {
    UdfTwlTimerWheelType level; /* 时间轮级别 */
    uint32_t interval;          /* 时间轮对应的时间间隔 */
} UdfTwlTimerTypeTransferMap;

uint32_t UdfTwlTimerCalcAdjustInterval(UdfTwlTimerInst *timerInst, UdfTwlTimerWheelType level, uint64_t startTime)
{
    uint32_t i;
    uint32_t interval = 0;
    uint32_t basicInterval = UDF_TWL_TIMER_HOUR_INTERVAL;
    uint32_t precisionInterval = UDF_TWL_TIMER_HOUR_INTERVAL;
    char funcName[UDF_TML_TIMER_PROC_NAME_LEN] = {0};
    UdfTwlTimerTypeTransferMap transferMap[] = {
        {UDF_TWL_TIMER_WHEEL_MSEC, UDF_TWL_TIMER_MSEC_INTERVAL},
        {UDF_TWL_TIMER_WHEEL_SEC, UDF_TWL_TIMER_SEC_INTERVAL},
        {UDF_TWL_TIMER_WHEEL_MINUTE, UDF_TWL_TIMER_MINUTE_INTERVAL},
        {UDF_TWL_TIMER_WHEEL_HOUR, UDF_TWL_TIMER_HOUR_INTERVAL},
    };

    if (timerInst->dbgInfo.lastSysTime == 0) {
        return 0;
    }

    if (level >= UDF_TWL_TIMER_WHEEL_BUTT) {
        return 0;
    }

    for (i = 0; i < sizeof(transferMap) / sizeof(UdfTwlTimerTypeTransferMap); i++) {
        if (transferMap[i].level == level) {
            basicInterval = transferMap[i].interval;
        }
        if (transferMap[i].level == timerInst->precision) {
            precisionInterval = transferMap[i].interval;
        }
    }

    interval = startTime - timerInst->dbgInfo.lastSysTime;
    if (interval < timerInst->realInterval) {
        return (timerInst->realInterval - interval) / basicInterval;
    }

    if (interval >= (timerInst->realInterval + (UDF_TWL_PRECISE_FACTOR5 * precisionInterval))) {
        DbgGetFuncName(timerInst->proc, funcName, UDF_TML_TIMER_PROC_NAME_LEN);
        LOG_INNER_WARN("Timer proc trigger interval warning. (lastTriggerTime = %" PRIu64 ","
                       "triggerTime = %" PRIu64 ", func = %s)",
            timerInst->dbgInfo.lastSysTime, startTime, funcName);
    }

    return 0;
}

void UdfTwlTimerExecuteProc(UdfTwlTimerWheelCtrl *wheelCtrl, UdfTwlTimerWheel *timerWheel, UdfTwlTimerWheelType level,
    uint64_t val)
{
    (void)val;
    uint32_t adjustNum = 0;
    UdfTwlTimerInst *dfsTimer = NULL;
    VOS_LIST_HEAD_S *dllNode = NULL;
    VOS_LIST_HEAD_S *nextDllNode = NULL;
    uint64_t startTime = 0;
    uint64_t endTime = 0;
    char timeOutTime[BASE_TIMER_LAST_PROC_TIME_LEN] = {0};
    uint32_t curSlot = timerWheel->cur_slot;

    VOS_LIST_FOR_EACH_ITEM_SAFE(dllNode, nextDllNode, &(timerWheel->listHead[curSlot]))
    {
        dfsTimer = VOS_LIST_ENTRY(dllNode, UdfTwlTimerInst, node);
        if (dfsTimer->rotation != 0) {
            dfsTimer->rotation--;
            continue;
        }

        if ((dfsTimer->proc != NULL) && (dfsTimer->status != UDF_TWLTMR_INST_STATUS_DEL)) {
            UdfTimeStrGet(timeOutTime, BASE_TIMER_LAST_PROC_TIME_LEN);
            (void)SystimeGetMilliSec(&startTime);
            adjustNum = UdfTwlTimerCalcAdjustInterval(dfsTimer, level, startTime);
            if (adjustNum > 0) {
                UdfTimerInstMoveNode(timerWheel, dfsTimer);
                (void)UdfTimerInstAddNode(wheelCtrl, level, dfsTimer, adjustNum, UDF_TWL_TIMER_INST_WAIT_EXECUTE);
                continue;
            }

            dfsTimer->begingSchedule = true;
            (void)dfsTimer->proc(dfsTimer->usrData, NULL);
            dfsTimer->begingSchedule = false;
            (void)SystimeGetMilliSec(&endTime);
        }

        UdfTwlTimerUpdateDbgInfo(dfsTimer, startTime, endTime, timeOutTime);
        UdfTwlTimerReloadInst(wheelCtrl, dfsTimer, timerWheel);
    }
}

UdfTwlTimerWheelType UdfTwlTimerGetHigherLevel(UdfTwlTimerWheelType level)
{
    uint32_t i;
    UdfTwlTimerWheelType wheelList[] = {
        UDF_TWL_TIMER_WHEEL_MSEC, UDF_TWL_TIMER_WHEEL_SEC, UDF_TWL_TIMER_WHEEL_MINUTE, UDF_TWL_TIMER_WHEEL_HOUR};

    if (level == UDF_TWL_TIMER_WHEEL_HOUR) {
        return UDF_TWL_TIMER_WHEEL_BUTT;
    }

    for (i = 0; i < (sizeof(wheelList) / sizeof(UdfTwlTimerWheelType)) - 1; i++) {
        if (wheelList[i] == level) {
            return wheelList[i + 1];
        }
    }

    return UDF_TWL_TIMER_WHEEL_BUTT;
}

UdfTwlTimerWheelType UdfTwlTimerGetLowerLevel(UdfTwlTimerWheelType level)
{
    uint32_t i;
    UdfTwlTimerWheelType wheelList[] = {
        UDF_TWL_TIMER_WHEEL_MSEC, UDF_TWL_TIMER_WHEEL_SEC, UDF_TWL_TIMER_WHEEL_MINUTE, UDF_TWL_TIMER_WHEEL_HOUR};

    if (level == UDF_TWL_TIMER_WHEEL_MSEC) {
        return UDF_TWL_TIMER_WHEEL_MSEC;
    }

    for (i = 1; i < (sizeof(wheelList) / sizeof(UdfTwlTimerWheelType)); i++) {
        if (wheelList[i] == level) {
            return wheelList[i - 1];
        }
    }

    return UDF_TWL_TIMER_WHEEL_BUTT;
}

void UdfTwlTimerHigherLevelProc(UdfTwlTimerWheelCtrl *wheelCtrl, UdfTwlTimerWheel *timerWheel,
    UdfTwlTimerWheelType level)
{
    uint32_t adjustNum = 0;
    uint64_t startTime = 0;
    UdfTwlTimerInst *dfsTimer = NULL;
    VOS_LIST_HEAD_S *dllNode = NULL;
    VOS_LIST_HEAD_S *nextDllNode = NULL;
    UdfTwlTimerWheelType nextLevel;
    uint32_t curSlot = timerWheel->cur_slot;

    VOS_LIST_FOR_EACH_ITEM_SAFE(dllNode, nextDllNode, &(timerWheel->listHead[curSlot]))
    {
        dfsTimer = VOS_LIST_ENTRY(dllNode, UdfTwlTimerInst, node);
        if (dfsTimer->rotation != 0) {
            dfsTimer->rotation--;
            continue;
        }

        (void)SystimeGetMilliSec(&startTime);
        UdfTimerInstMoveNode(timerWheel, dfsTimer);
        adjustNum = UdfTwlTimerCalcAdjustInterval(dfsTimer, level, startTime);
        if (adjustNum > 0) {
            (void)UdfTimerInstAddNode(wheelCtrl, level, dfsTimer, adjustNum, UDF_TWL_TIMER_INST_WAIT_EXECUTE);
        } else {
            nextLevel = UdfTwlTimerGetLowerLevel(level);
            adjustNum = UdfTwlTimerCalcAdjustInterval(dfsTimer, nextLevel, startTime);
            (void)UdfTimerInstAddNode(wheelCtrl, nextLevel, dfsTimer, adjustNum, UDF_TWL_TIMER_INST_WAIT_EXECUTE);
        }
    }

    return;
}

void UdfTwlTimerProcess(UdfTwlTimerWheelCtrl *wheelCtrl, UdfTwlTimerWheelType level, uint64_t val)
{
    uint64_t i;
    UdfTwlTimerWheel *timerWheel = NULL;

    timerWheel = UdfTwlTimerGetRunningWheel(wheelCtrl, level);
    if (timerWheel == NULL) {
        return;
    }

    for (i = 0; i < val; i++) {
        if ((uint32_t)timerWheel->cur_slot >= timerWheel->slot_num - 1) {
            timerWheel->cur_slot = 0;
            UdfTwlTimerProcess(wheelCtrl, UdfTwlTimerGetHigherLevel(level), 1);
        } else {
            timerWheel->cur_slot = timerWheel->cur_slot + 1;
        }

        if (level != wheelCtrl->execLevel) {
            UdfTwlTimerHigherLevelProc(wheelCtrl, timerWheel, level);
            return;
        }

        UdfTwlTimerExecuteProc(wheelCtrl, timerWheel, level, val);
    }

    return;
}

uint32_t UdfTwlTimerEvtProcess(BaseTimerHandle handle, uint64_t val, void *param)
{
    (void)handle;
    UdfTwlTimerWheelCtrl *wheelCtrl = (UdfTwlTimerWheelCtrl *)param;
    if ((wheelCtrl == NULL) || (val == 0)) {
        LOG_INNER_ERR("Twl timer proc: count:%" PRIu64 " not correct", val);
        return BASE_TIMER_ERRNO_INVAL;
    }

    UdfTwlTimerProcess(wheelCtrl, wheelCtrl->execLevel, val);
    return UDF_OK;
}

BaseTimerAttr UdfTwlTimerCreateBaseAttr(TimerAppInfo *appInfo, char *name)
{
    uint32_t ret;
    BaseTimerAttr attr;

    attr = BASE_TimerCreateAttr(appInfo);
    if (attr == NULL) {
        return NULL;
    }

    ret = BASE_TimerSetAttr(attr, BASE_TIMER_ATTR_NAME, (void *)name, strlen(name));
    if (ret != BASE_TIMER_ERRNO_OK) {
        BASE_TimerDestroyAttr(appInfo, attr);
        return NULL;
    }

    ret = BASE_TimerSetAttr(attr, BASE_TIMER_ATTR_USRPARAM_DESTRUCTOR, (void *)(uintptr_t)UdfTwlTimerDestoryWheel,
        sizeof(void *));
    if (ret != BASE_TIMER_ERRNO_OK) {
        BASE_TimerDestroyAttr(appInfo, attr);
        return NULL;
    }

    return attr;
}

uint32_t UdfTwlTimerCreateRelTmr(UdfTwlTimerWheelCtrl *wheelCtrl, uint32_t timeOutMs)
{
    BaseTimerUsrCb usrCb;
    BaseTimerAttr baseAttr = NULL;
    BaseTimerHandle tmrHandle = NULL;

    if ((wheelCtrl->name[0] != '\0')) {
        baseAttr = UdfTwlTimerCreateBaseAttr(&wheelCtrl->appInfo, wheelCtrl->name);
        if (baseAttr == NULL) {
            return UDF_ERROR;
        }
    }

    usrCb.proc = UdfTwlTimerEvtProcess;
    usrCb.param = (void *)wheelCtrl;

    tmrHandle = BASE_TimerCreate(&wheelCtrl->appInfo, &usrCb, timeOutMs, baseAttr);
    if (tmrHandle == NULL) {
        return UDF_ERROR;
    }

    wheelCtrl->timerHd = tmrHandle;
    return UDF_OK;
}

uint32_t UdfCreateTwlTimerWheel(UdfTwlTimerWheelCtrl *wheelCtrl, UdfTimerInitParam *initParam)
{
    wheelCtrl->magic = UDF_TWL_TIMER_MAGIC;
    wheelCtrl->appInfo.appHandle = initParam->appHandle;
    wheelCtrl->appInfo.appCid = initParam->appCid;
    wheelCtrl->appInfo.memHandle = initParam->memHandle;
    wheelCtrl->appInfo.epollCtlCB = initParam->epollCtlCB;
    wheelCtrl->appInfo.mallocCB = initParam->mallocCB;
    wheelCtrl->appInfo.freeCB = initParam->freeCB;

    wheelCtrl->execLevel = UDF_TWL_TIMER_WHEEL_BUTT;
    VOS_ListInit(&(wheelCtrl->timerlist));
    errno_t errNo;
    if (initParam->timerInstName[0] == '\0') {
        errNo = strcpy_s(wheelCtrl->name, UDF_TWL_TMR_NAME_LEN, UDF_TWL_TIMER_DEFAULT_NAME);
    } else {
        errNo = strcpy_s(wheelCtrl->name, UDF_TWL_TMR_NAME_LEN, initParam->timerInstName);
    }
    if (errNo != EOK) {
        return UDF_ERROR;
    }
    uint32_t ret = UdfTwlTimerCreateRelTmr(wheelCtrl, UDF_TWL_TIMER_HOUR_INTERVAL);
    if (ret != UDF_OK) {
        LOG_INNER_ERR("Udf create basic timer unsuccessfully.");
        return ret;
    }

    return UDF_OK;
}

bool UdfTwlTimerGetDeployInfo(UdfTwlTimerWheelCtrl *twlNode, void *appHandle)
{
    if (twlNode == NULL || twlNode->magic != UDF_TWL_TIMER_MAGIC) {
        return false;
    }

    BaseTimerInfo *tmrInfo = (BaseTimerInfo *)twlNode->timerHd;
    if ((tmrInfo == NULL) || (tmrInfo->delTag == BASE_TIMER_DEL_TAG_TRUE)) {
        return false;
    }

    if (twlNode->appInfo.appHandle != appHandle) {
        return false;
    }

    return true;
}

uint32_t UdfTimerCheckInitParam(UdfTimerInitParam *initParam)
{
    if (initParam->appHandle == NULL) {
        return UDF_ERROR;
    }

    if ((initParam->mallocCB == NULL) || (initParam->freeCB == NULL)) {
        return UDF_ERROR;
    }

    if (initParam->epollCtlCB == NULL) {
        return UDF_ERROR;
    }

    return UDF_OK;
}

uint32_t UdfTwlTimerCreate(UdfTimerInitParam *initParam, UdfTwlTimerHandle *handle)
{
    uint32_t ret = UDF_OK;
    UdfTwlTimerWheelCtrl *wheelCtrl = NULL;
    if (initParam == NULL) {
        LOG_INNER_ERR("Twl timer create, initParam is NULL!");
        return UDF_ERROR;
    }

    if (UdfTimerCheckInitParam(initParam) != UDF_OK) {
        return UDF_ERROR;
    }

    do {
        wheelCtrl = initParam->mallocCB(initParam->appHandle, sizeof(UdfTwlTimerWheelCtrl), TIMER_FNAME, TIMER_FLINE);
        if (wheelCtrl == NULL) {
            return UDF_ERROR;
        }

        (void)memset_s(wheelCtrl, sizeof(UdfTwlTimerWheelCtrl), 0, sizeof(UdfTwlTimerWheelCtrl));
        ret = UdfCreateTwlTimerWheel(wheelCtrl, initParam);
        if (ret != UDF_OK) {
            initParam->freeCB(initParam->appHandle, wheelCtrl, TIMER_FNAME, TIMER_FLINE);
            return UDF_ERROR;
        }

        *handle = wheelCtrl;
    } while (0);

    if (ret == UDF_OK) {
        wheelCtrl->creatCnt++;
    }

    return ret;
}

void UdfTwlTimerDestroy(UdfTwlTimerHandle handle)
{
    UdfTwlTimerWheelCtrl *wheelCtrl = (UdfTwlTimerWheelCtrl *)handle;
    if ((wheelCtrl == NULL) || (wheelCtrl->magic != UDF_TWL_TIMER_MAGIC)) {
        LOG_INNER_ERR("Twl timer destroy, parameter invalid.");
        return;
    }

    if (wheelCtrl->creatCnt > 0) {
        wheelCtrl->creatCnt--;
    }

    BaseTimerInfo *tmrInfo = (BaseTimerInfo *)wheelCtrl->timerHd;
    if ((wheelCtrl->creatCnt == 0) && (tmrInfo != NULL) && (tmrInfo->delTag != BASE_TIMER_DEL_TAG_TRUE)) {
        BASE_TimerDestroy(wheelCtrl->timerHd);
        BaseTimerDestroy(tmrInfo); // 注意：后面就不能再访问wheelCtrl和tmrInfo了，因为已经释放
        wheelCtrl = NULL;
        tmrInfo = NULL;
    }

    return;
}

UdfTwlTimerInst *UdfTimerInstNodeCreate(UdfTwlTimerWheelCtrl *wheelCtrl, UdfTmrInstParam *timerParam,
    UdfTmrInstStatus status, TimerDestructProc func)
{
    UdfTwlTimerInst *timerInst = NULL;
    uint32_t ret;
    timerInst = (UdfTwlTimerInst *)Timer_MemAlloc(&wheelCtrl->appInfo, sizeof(UdfTwlTimerInst));
    if (timerInst == NULL) {
        return NULL;
    }

    (void)memset_s(timerInst, sizeof(UdfTwlTimerInst), 0, sizeof(UdfTwlTimerInst));

    timerInst->magic = UDF_TWL_TIMER_INST_MAGIC;
    timerInst->tmrMode = timerParam->tmrMode;
    timerInst->usrData = timerParam->usrData;
    timerInst->proc = timerParam->callBack;
    timerInst->resizeInterval = UDF_TWLTMR_INTERVAL_INVALID;
    timerInst->realInterval = timerParam->interval;
    timerInst->status = status;
    timerInst->destructProc = func;
    timerInst->WheelHandle = wheelCtrl;
    errno_t errNo;
    if (timerParam->tmrName[0] == '\0') {
        errNo = strcpy_s(timerInst->timername, UDF_TWL_TMR_NAME_LEN, UDF_TWL_TIMER_INST_DEFAULT_NAME);
    } else {
        errNo = strcpy_s(timerInst->timername, UDF_TWL_TMR_NAME_LEN, timerParam->tmrName);
    }
    if (errNo != EOK) {
        Timer_MemFree(&wheelCtrl->appInfo, timerInst);
        return NULL;
    }
    ret = UdfTimerInstCalcPrecision(wheelCtrl, timerInst, timerInst->realInterval);
    if (ret != UDF_OK) {
        Timer_MemFree(&wheelCtrl->appInfo, timerInst);
        return NULL;
    }

    VOS_ListInit(&timerInst->nodeForMT);
    VOS_ListAdd(&(timerInst->nodeForMT), &(wheelCtrl->timerlist));
    return timerInst;
}

uint32_t UdfTimerInstCheckParam(UdfTwlTimerWheelCtrl *wheelCtrl, UdfTmrInstParam *timerParam,
    UdfTmrInstHandle *tmrInstHandle)
{
    if ((wheelCtrl == NULL) || (wheelCtrl->magic != UDF_TWL_TIMER_MAGIC)) {
        return BASE_TIMER_ERRNO_INVAL;
    }

    if ((timerParam == NULL) || (tmrInstHandle == NULL)) {
        return BASE_TIMER_ERRNO_INVAL;
    }

    if ((timerParam->tmrMode != UDF_TIMER_MODE_ONE_SHOT) && (timerParam->tmrMode != UDF_TIMER_MODE_PERIOD)) {
        return BASE_TIMER_ERRNO_INVAL;
    }

    if ((timerParam->callBack == NULL) || (timerParam->interval < UDF_TWL_TIMER_MSEC_INTERVAL)) {
        return BASE_TIMER_ERRNO_INVAL;
    }

    return UDF_OK;
}

uint32_t UdfTimerInstCreate(UdfTwlTimerHandle handle, UdfTmrInstParam *timerParam, UdfTmrInstHandle *tmrInstHandle)
{
    uint32_t ret;
    UdfTwlTimerWheelCtrl *wheelCtrl = (UdfTwlTimerWheelCtrl *)handle;
    UdfTwlTimerInst *timerInst = NULL;

    ret = UdfTimerInstCheckParam(wheelCtrl, timerParam, tmrInstHandle);
    if (ret != UDF_OK) {
        return ret;
    }

    timerInst = UdfTimerInstNodeCreate(wheelCtrl, timerParam, UDF_TWLTMR_INST_STATUS_RUN, NULL);
    if (timerInst == NULL) {
        return BASE_TIMER_ERRNO_ERR;
    }

    *tmrInstHandle = timerInst;
    return UDF_OK;
}

uint32_t UdfTimerInstCreateNotStart(UdfTwlTimerHandle handle, TimerDestructProc func, UdfTmrInstParam *timerParam,
    UdfTmrInstHandle *tmrInstHandle)
{
    uint32_t ret;
    UdfTwlTimerWheelCtrl *wheelCtrl = (UdfTwlTimerWheelCtrl *)handle;
    UdfTwlTimerInst *timerInst = NULL;

    ret = UdfTimerInstCheckParam(wheelCtrl, timerParam, tmrInstHandle);
    if (ret != UDF_OK) {
        return ret;
    }

    timerInst = UdfTimerInstNodeCreate(wheelCtrl, timerParam, UDF_TWLTMR_INST_STATUS_STOP, func);
    if (timerInst == NULL) {
        return BASE_TIMER_ERRNO_NOMEM;
    }

    *tmrInstHandle = timerInst;
    return UDF_OK;
}

bool UdfTimerInstIsValid(UdfTwlTimerInst *timerInst)
{
    if ((timerInst == NULL) || (timerInst->magic != UDF_TWL_TIMER_INST_MAGIC) ||
        (timerInst->status == UDF_TWLTMR_INST_STATUS_DEL)) {
        return false;
    }

    return true;
}

void UdfTimerInstDelete(UdfTmrInstHandle tmrInst)
{
    UdfTwlTimerWheelCtrl *wheelCtrl = NULL;
    UdfTwlTimerInst *timerInst = (UdfTwlTimerInst *)tmrInst;
    if (!UdfTimerInstIsValid(timerInst)) {
        LOG_INNER_ERR("Twl timer inst destroy, inst handle invalid.");
        return;
    }

    wheelCtrl = timerInst->WheelHandle;
    timerInst->status = UDF_TWLTMR_INST_STATUS_DEL;
    if (timerInst->begingSchedule) {
        return;
    }

    UdfTimerInstDelNode(wheelCtrl, timerInst->tmrWheel, timerInst);

    return;
}

uint32_t UdfTimerInstResize(UdfTmrInstHandle tmrInst, uint32_t timeOutMs)
{
    UdfTwlTimerWheelCtrl *wheelCtrl = NULL;
    UdfTwlTimerInst *timerInst = (UdfTwlTimerInst *)tmrInst;
    if (!UdfTimerInstIsValid(timerInst)) {
        LOG_INNER_ERR("Twl timer inst resize, inst handle invalid.");
        return BASE_TIMER_ERRNO_INVAL;
    }

    /* 单次定时器不支持重调超时时间 */
    if (timerInst->tmrMode == UDF_TIMER_MODE_ONE_SHOT) {
        return BASE_TIMER_ERRNO_NOT_SUPPORT;
    }

    wheelCtrl = timerInst->WheelHandle;
    timerInst->resizeInterval = timeOutMs;
    timerInst->status = UDF_TWLTMR_INST_STATUS_RUN;

    VOS_ListRemove(&(timerInst->nodeForMT));
    VOS_ListAdd(&(timerInst->nodeForMT), &(wheelCtrl->timerlist));
    UdfTimerInstMoveNode(timerInst->tmrWheel, timerInst);
    uint32_t ret = UdfTimerInstCalcPrecision(wheelCtrl, timerInst, timerInst->resizeInterval);
    if (ret != UDF_OK) {
        LOG_INNER_ERR("Udf timer instance timeout interval resize unsuccess. (0x%x).", ret);
        return UDF_ERROR;
    }

    return ret;
}

uint32_t UdfTimerInstGetRemainTime(UdfTmrInstHandle tmrInst, uint32_t *remainMs)
{
    uint32_t ret;
    UdfTwlTimerWheelCtrl *wheelCtrl = NULL;
    UdfTwlTimerInst *timerInst = (UdfTwlTimerInst *)tmrInst;
    if (!UdfTimerInstIsValid(timerInst)) {
        LOG_INNER_ERR("Twl timer inst get remain time, inst handle invalid.");
        return BASE_TIMER_ERRNO_INVAL;
    }

    wheelCtrl = timerInst->WheelHandle;
    ret = UdfTwlTimerCalcRemainTime(wheelCtrl, timerInst, remainMs);
    return ret;
}

typedef uint32_t (*UdfTwlTimerInstTravFunc)(void *paraIn, UdfTwlTimerInst *timerInst);

uint32_t UdfTwlTimerInstDbgDataScan(UdfTwlTimerWheelCtrl *wheelCtrl, UdfTwlTimerInstTravFunc trav, void *paraIn)
{
    uint32_t ret;
    VOS_LIST_HEAD_S *item = NULL;
    UdfTwlTimerInst *timerInst = NULL;

    VOS_LIST_FOR_EACH_ITEM(item, &wheelCtrl->timerlist)
    {
        timerInst = VOS_LIST_ENTRY(item, UdfTwlTimerInst, nodeForMT);
        ret = trav(paraIn, timerInst);
        if (ret != UDF_OK) {
            LOG_INNER_ERR("Scan twl timer inst dbg info, traverse data entry unsuccessfully with 0x%x.", ret);
            return ret;
        }
    }

    return UDF_OK;
}

uint32_t UdfTwlTimerInstDbgDataTrav(void *paraIn, UdfTwlTimerInst *timerInst)
{
    (void)paraIn;
    char funcName[UDF_TML_TIMER_PROC_NAME_LEN] = {0};

    if (timerInst == NULL) {
        return BASE_TIMER_ERRNO_INVAL;
    }

    DbgGetFuncName(timerInst->proc, funcName, UDF_TML_TIMER_PROC_NAME_LEN);
    char *tmrMode = (timerInst->tmrMode == UDF_TIMER_MODE_ONE_SHOT) ? "ONE_SHOT" : "PERIOD";

    UDF_CMD_OUTPUT("%-16s  %-30s  %16u  %-8s %26s  %16u  %16u  %16u    %" PRIu64 "\n", timerInst->timername, funcName,
        timerInst->realInterval, tmrMode, timerInst->dbgInfo.lastTime, timerInst->dbgInfo.maxProcTime,
        timerInst->dbgInfo.maxInterval, timerInst->dbgInfo.minInterval, timerInst->dbgInfo.timeOutCount);
    return UDF_OK;
}

char *UdfTwlTimerGetPrecisionStr(UdfTwlTimerWheelType execLevel)
{
    char *taskStatus[] = {"MILLISECOND", "SECOND", "MINUTE", "HOUR", "NONE"};
    return taskStatus[execLevel];
}

uint32_t UdfTwlTimerGetTimerInfo(UdfTwlTimerWheelCtrl *twlNode)
{
    if (twlNode == NULL || twlNode->magic != UDF_TWL_TIMER_MAGIC) {
        return BASE_TIMER_ERRNO_INVAL;
    }

    BaseTimerInfo *tmrInfo = (BaseTimerInfo *)twlNode->timerHd;
    if (tmrInfo == NULL) {
        return BASE_TIMER_ERRNO_INVAL;
    }

    UDF_CMD_OUTPUT("%.*s\n", UDF_TWL_OUTSIZE_40, UDF_CMD_SPLIT_LINE);
    UDF_CMD_OUTPUT("%-16s  %-16s\n", "Twl Timer:", twlNode->name);
    UDF_CMD_OUTPUT("%.*s\n", UDF_TWL_OUTSIZE_40, UDF_CMD_SPLIT_LINE); /* 24 是对应显示头的输出长度 */
    UDF_CMD_OUTPUT("%-16s  %10d\n", "TimerFd:", tmrInfo->tmrFd);
    UDF_CMD_OUTPUT("%-16s  %-16s\n", "Precision:", UdfTwlTimerGetPrecisionStr(twlNode->execLevel));

    UDF_CMD_OUTPUT("%.*s\n", UDF_TWL_OUTSIZE_120, UDF_CMD_SPLIT_LINE);
    UDF_CMD_OUTPUT("%-16s  %-30s  %16s  %-8s  %26s  %16s  %16s  %16s  %-12s\n", "Name", "UsrHook", "Interval(ms)",
        "Mode", "LastTimeOutTime", "MaxProcTime(ms)", "MaxInterval(ms)", "MinInterval(ms)", "TimeOutCount");
    UDF_CMD_OUTPUT("%.*s\n", UDF_TWL_OUTSIZE_120, UDF_CMD_SPLIT_LINE);

    return UdfTwlTimerInstDbgDataScan(twlNode, UdfTwlTimerInstDbgDataTrav, NULL);
}

typedef struct {
    uint32_t totalNum;
    uint32_t iterateCnt;
    UdfTimerMntInfo *tmrInfo;
} TimerIterateData;

typedef struct {
    BaseTimerInfo *tmrInfo;
    void *usrData;
} UdfTwlTimerInfo;

uint32_t UdfTwlTimerGetInstInfo(void *paraIn, UdfTwlTimerInst *timerInst)
{
    UdfTwlTimerInfo *timerInfo = (UdfTwlTimerInfo *)paraIn;
    if (timerInfo == NULL || timerInst == NULL) {
        return BASE_TIMER_ERRNO_INVAL;
    }

    BaseTimerInfo *tmrInfo = timerInfo->tmrInfo;
    if (tmrInfo == NULL) {
        return BASE_TIMER_ERRNO_INVAL;
    }

    TimerIterateData *data = (TimerIterateData *)timerInfo->usrData;
    if (data->iterateCnt >= data->totalNum) {
        return UDF_OK;
    }

    UdfTimerMntInfo *tmrMntInfo = &data->tmrInfo[data->iterateCnt];

    errno_t errNo = strcpy_s(tmrMntInfo->lastTime, UDF_TIMER_LAST_PROC_TIME_LEN_MAX, timerInst->dbgInfo.lastTime);
    if (errNo != EOK) {
        return UDF_ERROR;
    }
    errNo = strcpy_s(tmrMntInfo->timerName, UDF_TIMER_NAME_LEN_MAX, timerInst->timername);
    if (errNo != EOK) {
        return UDF_ERROR;
    }
    tmrMntInfo->handle = (void *)tmrInfo;
    tmrMntInfo->interval = timerInst->realInterval;
    tmrMntInfo->status = (UdfTimerStatus)tmrInfo->mntInfo.status;
    tmrMntInfo->maxProcTime = timerInst->dbgInfo.maxProcTime;
    tmrMntInfo->timeoutCnt = timerInst->dbgInfo.timeOutCount;
    tmrMntInfo->mode = timerInst->tmrMode;
    char funcName[UDF_TIMER_USR_FUNC_NAME_MAX] = {0};
    DbgGetFuncName(timerInst->proc, funcName, UDF_TIMER_USR_FUNC_NAME_MAX);
    errNo = strcpy_s(tmrMntInfo->usrFunc, UDF_TIMER_USR_FUNC_NAME_MAX, funcName);
    if (errNo != EOK) {
        return UDF_ERROR;
    }
    data->iterateCnt++;

    return UDF_OK;
}
void *UdfTwlTimerGetInstUsrData(UdfTmrInstHandle tmrInstHdl)
{
    if (tmrInstHdl == NULL) {
        return NULL;
    }

    UdfTwlTimerInst *timerInst = (UdfTwlTimerInst *)tmrInstHdl;
    if (timerInst->magic != UDF_TWL_TIMER_INST_MAGIC) {
        return NULL;
    }

    return timerInst->usrData;
}

uint32_t UdfTwlTimerRunningInstGetCnt(UdfTwlTimerHandle timerHandle)
{
    uint32_t cnt = 0;
    VOS_LIST_HEAD_S *item = NULL;

    if (timerHandle == NULL) {
        return 0;
    }
    UdfTwlTimerWheelCtrl *wheelCtrl = (UdfTwlTimerWheelCtrl *)timerHandle;

    if (wheelCtrl->magic != UDF_TWL_TIMER_MAGIC) {
        return 0;
    }

    VOS_LIST_FOR_EACH_ITEM(item, &wheelCtrl->timerlist)
    {
        UdfTwlTimerInst *timerInst = VOS_LIST_ENTRY(item, UdfTwlTimerInst, nodeForMT);
        if ((timerInst != NULL) && (timerInst->status == UDF_TWLTMR_INST_STATUS_RUN)) {
            cnt++;
        }
    }

    return cnt;
}

#ifdef __cplusplus
}
#endif
