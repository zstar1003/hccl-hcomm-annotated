/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "udf_base_timer.h"

#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <inttypes.h>
#include <string.h>
#include <sys/epoll.h>
#include <errno.h>
#include <fcntl.h>
#include "securec.h"
#include "udf_timer_adapt.h"

#ifdef __cplusplus
extern "C" {
#endif

#define BASE_TIMER_NAME_LEN_MAX 16

#define BASE_TIMER_DEFAULT_NAME "udfBaseTimer"

#define BASE_TIMER_ATTR_MAGIC 0xFFFFFFF1
#define BASE_TIMER_HANDLE_MAGIC 0xFFFFFFF2

#define BASE_TIMER_WARNING_PROC_THRESHOLD 400

#pragma pack(4)

typedef struct {
    uint32_t magic;
    BaseTimerParam param;
} BaseTimerParamAttr;

typedef struct {
    VOS_LIST_HEAD_S list; /* 定时器并发修改保护 */
} BaseTimerCtrl;

#pragma pack()

void BaseTimerUpdateMntInfo(BaseTimerInfo *tmrInfo, uint32_t procTime, char *timeOutTime);
void BaseTimerDestroy(BaseTimerInfo *tmrInfo);

void BaseTimerDestroyInfo(BaseTimerInfo *tmrInfo)
{
    if (tmrInfo != NULL) {
        if (tmrInfo->optParam != NULL) {
            Timer_MemFree(&tmrInfo->appInfo, tmrInfo->optParam);
        }
        Timer_MemFree(&tmrInfo->appInfo, tmrInfo);
    }
    return;
}

BaseTimerInfo *BaseTimerCreateInfo(TimerAppInfo *appInfo, BaseTimerUsrCb *usrCb, uint32_t timeoutMs,
    BaseTimerParamAttr *attr)
{
    BaseTimerParam *optParam = NULL;
    BaseTimerParam *param = (attr == NULL ? NULL : &(attr->param));

    BaseTimerInfo *tmrInfo = appInfo->mallocCB(appInfo->memHandle, sizeof(BaseTimerInfo), TIMER_FNAME, TIMER_FLINE);
    if (tmrInfo == NULL) {
        return NULL;
    }
    (void)memset_s(tmrInfo, sizeof(BaseTimerInfo), 0, sizeof(BaseTimerInfo));

    tmrInfo->magic = BASE_TIMER_HANDLE_MAGIC;
    tmrInfo->tmrFd = -1;
    BASE_TimerCopyAppInfo(&tmrInfo->appInfo, appInfo);
    (void)memcpy_s(&tmrInfo->usrCb, sizeof(BaseTimerUsrCb), usrCb, sizeof(BaseTimerUsrCb));
    tmrInfo->timeoutMs = timeoutMs;
    tmrInfo->delTag = BASE_TIMER_DEL_TAG_FALSE;
    tmrInfo->optParam = NULL;

    if (param != NULL) {
        optParam = Timer_MemAlloc(&tmrInfo->appInfo, sizeof(BaseTimerParam));
        if (optParam == NULL) {
            appInfo->freeCB(&appInfo->appHandle, tmrInfo, TIMER_FNAME, TIMER_FLINE);
            return NULL;
        }
        (void)memcpy_s(optParam, sizeof(BaseTimerParam), param, sizeof(BaseTimerParam));

        tmrInfo->optParam = optParam;
    }

    return tmrInfo;
}
void BaseTimerEventProc(BaseTimerInfo *tmrInfo, uint64_t readValue)
{
    uint64_t startTime = 0;
    uint64_t endTime = 0;
    char timeOutTime[BASE_TIMER_LAST_PROC_TIME_LEN] = {0};
    uint32_t usrRet = 0;

    BaseTimerProc proc = tmrInfo->usrCb.proc;
    if (proc == NULL) {
        return;
    }

    uint32_t delTag = tmrInfo->delTag;
    if (delTag != BASE_TIMER_DEL_TAG_TRUE) {
        void *param = tmrInfo->usrCb.param;
        UdfTimeStrGet(timeOutTime, BASE_TIMER_LAST_PROC_TIME_LEN);
        (void)SystimeGetMilliSec(&startTime);
        usrRet = proc(tmrInfo, readValue, param);
        (void)SystimeGetMilliSec(&endTime);
    }

    if (endTime - startTime >= BASE_TIMER_WARNING_PROC_THRESHOLD) {
        char funcName[UDF_TASK_PROC_NAME_MAX_LEN] = {0};
        DbgGetFuncName(proc, funcName, UDF_TASK_PROC_NAME_MAX_LEN);
        LOG_INNER_WARN("Timer proc threshold warning. (timerfd = %d, startTime = %" PRIu64 ","
                       "procTime =%" PRIu64 ", func = %s)",
            tmrInfo->tmrFd, startTime, endTime - startTime, funcName);
    }

    BaseTimerUpdateMntInfo(tmrInfo, endTime - startTime, timeOutTime);

    BaseTimerMode tmrMode = tmrInfo->optParam == NULL ? BASE_TIMER_MODE_PERIOD : tmrInfo->optParam->mode;
    if ((tmrMode == BASE_TIMER_MODE_ONE_SHOT) || (delTag == BASE_TIMER_DEL_TAG_TRUE)) {
        BaseTimerDestroy(tmrInfo);
    }

    if (usrRet != 0) {
        LOG_INNER_ERR("Timer usr hook proc course mistake. (ret = %#x)", usrRet);
    }

    return;
}

void BaseTimerTriggerProc(struct epoll_event *event, void *schParam)
{
    int readCnt;
    BaseTimerInfo *tmrInfo = (BaseTimerInfo *)schParam;

    /* 定时器节点删除在此函数上下文，此处定时器肯定存在 */
    if (tmrInfo->magic != BASE_TIMER_HANDLE_MAGIC) {
        return;
    }

    if (event->data.fd != tmrInfo->tmrFd) {
        LOG_INNER_ERR("Timer fd event fd mismatch: input %d, expect %d.", event->data.fd, tmrInfo->tmrFd);
        return;
    }

    if ((event->events & EPOLLIN) != 0) {
        uint64_t readValue;
        readCnt = read(tmrInfo->tmrFd, &readValue, sizeof(uint64_t));
        if (readCnt != sizeof(uint64_t)) {
            return;
        }

        BaseTimerEventProc(tmrInfo, readValue);
    }

    return;
}

void BaseTimerDestroy(BaseTimerInfo *tmrInfo)
{
    uint32_t ret;
    void *usrParam = NULL;
    BaseTimerParamDestruct destruct = NULL;

    UdfTimerFdParams param;
    param.operate = EPOLL_CTL_DEL;
    param.timerFdProc = BaseTimerTriggerProc;
    param.timerEventParam = (void *)tmrInfo;
    param.event.events = EPOLLIN;
    param.event.data.ptr = NULL;
    param.event.data.fd = tmrInfo->tmrFd;

    ret = tmrInfo->appInfo.epollCtlCB(tmrInfo->appInfo.appHandle, tmrInfo->appInfo.appCid, tmrInfo->tmrFd, &param);
    if (ret != 0) {
        return;
    }

    if (tmrInfo->tmrFd > 0) {
        (void)close(tmrInfo->tmrFd);
        tmrInfo->tmrFd = -1;
    }

    if ((tmrInfo->optParam != NULL) && (tmrInfo->optParam->destructor != NULL)) {
        destruct = tmrInfo->optParam->destructor;
        usrParam = tmrInfo->usrCb.param;
    }

    tmrInfo->magic = 0;
    BaseTimerDestroyInfo(tmrInfo);

    if (destruct != NULL) { /* 用户析构函数锁外执行 */
        destruct(usrParam);
    }

    return;
}

void BaseTimerUpdateMntInfo(BaseTimerInfo *tmrInfo, uint32_t procTime, char *timeOutTime)
{
    if (procTime > tmrInfo->mntInfo.maxProcTime) {
        tmrInfo->mntInfo.maxProcTime = procTime;
    }

    errno_t errNo = strcpy_s(tmrInfo->mntInfo.lastTime, BASE_TIMER_LAST_PROC_TIME_LEN, timeOutTime);
    if (errNo != EOK) {
        return;
    }
    tmrInfo->mntInfo.timeOutCount++;

    return;
}

uint32_t BaseTimerDeploy(BaseTimerInfo *tmrInfo)
{
    UdfTimerFdParams param;
    param.operate = EPOLL_CTL_ADD;
    param.timerFdProc = BaseTimerTriggerProc;
    param.timerEventParam = (void *)tmrInfo;
    param.event.events = EPOLLIN;
    param.event.data.ptr = NULL;
    param.event.data.fd = tmrInfo->tmrFd;

    return tmrInfo->appInfo.epollCtlCB(tmrInfo->appInfo.memHandle, tmrInfo->appInfo.appCid, tmrInfo->tmrFd, &param);
}

uint32_t BaseTimerSetSch(BaseTimerInfo *tmrInfo)
{
    int tmrFd = timerfd_create(CLOCK_MONOTONIC, O_NONBLOCK);
    if (tmrFd < 0) {
        return BASE_TIMER_ERRNO_TIMER_CREATE_FAILED;
    }

    tmrInfo->tmrFd = tmrFd;
    uint32_t ret = BaseTimerDeploy(tmrInfo);
    if (ret != 0) {
        (void)close(tmrFd);
        LOG_INNER_ERR("Event deploy unsuccessfully. (ret = %#x)", ret);
        return ret;
    }

    return BASE_TIMER_ERRNO_OK;
}

uint32_t BaseTimerUpdateTime(BaseTimerInfo *timerInfo, uint32_t timeoutMs)
{
    int ret;
    struct itimerspec newValue = {{0}, {0}};

    newValue.it_value.tv_sec = (time_t)(timeoutMs / 1000ULL);                /* Ms(毫秒) / 1000 取得 s(秒) */
    newValue.it_value.tv_nsec = (int64_t)(timeoutMs % 1000ULL) * 1000000ULL; /* Ms(毫秒) % 1000 后乘 1000000 得到纳秒 */

    if ((timerInfo->optParam == NULL) || (timerInfo->optParam->mode == BASE_TIMER_MODE_PERIOD)) {
        newValue.it_interval.tv_sec = (time_t)(timeoutMs / 1000ULL); /* Ms(毫秒) / 1000 取得  s(秒) */
        /* Ms(毫秒) % 1000 后乘 1000000 得到纳秒 */
        newValue.it_interval.tv_nsec = (int64_t)(timeoutMs % 1000ULL) * 1000000ULL;
    }

    ret = timerfd_settime(timerInfo->tmrFd, 0, &newValue, NULL);
    if (ret != 0) {
        return BASE_TIMER_ERRNO_TIMER_SET_FAILED;
    }

    timerInfo->mntInfo.status = timeoutMs == 0 ? BASE_TIMER_STATUS_STOPED : BASE_TIMER_STATUS_ACTIVE;

    return BASE_TIMER_ERRNO_OK;
}

bool BaseTimerHandleValid(const BaseTimerInfo *tmrInfo)
{
    if ((tmrInfo == NULL) || (tmrInfo->magic != BASE_TIMER_HANDLE_MAGIC) ||
        (tmrInfo->delTag == BASE_TIMER_DEL_TAG_TRUE)) {
        return false;
    }

    return true;
}

uint32_t BaseTimerSetMode(BaseTimerParam *param, void *value, size_t len)
{
    if (len != sizeof(BaseTimerMode)) {
        return BASE_TIMER_ERRNO_PARAM_INVALID;
    }

    param->mode = *(BaseTimerMode *)value;

    return BASE_TIMER_ERRNO_OK;
}

uint32_t BaseTimerSetName(BaseTimerParam *param, void *value, size_t len)
{
    size_t lenVal;

    if (len >= BASE_TIMER_NAME_LEN_MAX) {
        return BASE_TIMER_ERRNO_PARAM_INVALID;
    }

    lenVal = strlen((char *)value);
    if (lenVal >= BASE_TIMER_NAME_LEN_MAX) {
        return BASE_TIMER_ERRNO_PARAM_INVALID;
    }

    if (strcpy_s(param->tmrName, BASE_TIMER_NAME_LEN_MAX, value) != EOK) {
        return BASE_TIMER_ERRNO_PARAM_INVALID;
    }

    return BASE_TIMER_ERRNO_OK;
}

uint32_t BaseTimerSetParamDestructor(BaseTimerParam *param, void *value, size_t len)
{
    if (len != sizeof(void *)) {
        return BASE_TIMER_ERRNO_PARAM_INVALID;
    }

    param->destructor = value;

    return BASE_TIMER_ERRNO_OK;
}

BaseTimerAttr BASE_TimerCreateAttr(TimerAppInfo *appInfo)
{
    BaseTimerParamAttr *attr = Timer_MemAlloc(appInfo, sizeof(BaseTimerParamAttr));
    if (attr == NULL) {
        return NULL;
    }
    attr->magic = BASE_TIMER_ATTR_MAGIC;
    attr->param.tmrName[0] = '\0';
    attr->param.mode = BASE_TIMER_MODE_PERIOD;
    attr->param.destructor = NULL;

    return attr;
}

void BASE_TimerDestroyAttr(TimerAppInfo *appInfo, BaseTimerAttr attr)
{
    BaseTimerParamAttr *attrParam = (BaseTimerParamAttr *)attr;
    if (attrParam == NULL) {
        return;
    }

    if (attrParam->magic != BASE_TIMER_ATTR_MAGIC) {
        LOG_INNER_ERR("Timer attr invalid. (magic = %#x)", attrParam->magic);
        return;
    }

    attrParam->magic = 0;
    if (appInfo == NULL) {
        return;
    }
    Timer_MemFree(appInfo, attr);

    return;
}

uint32_t BASE_TimerSetAttr(BaseTimerAttr attr, BaseTimerAttrType type, void *value, size_t len)
{
    BaseTimerParamAttr *attrParam = (BaseTimerParamAttr *)attr;
    uint32_t (*attrSetFunc[])(BaseTimerParam *param, void *value, size_t len) = {
        BaseTimerSetMode, BaseTimerSetName, BaseTimerSetParamDestructor};

    if ((attr == NULL) || (attrParam->magic != BASE_TIMER_ATTR_MAGIC)) {
        return BASE_TIMER_ERRNO_ATTR_INVALID;
    }

    if ((type >= BASE_TIMER_ATTR_BUTT) || (value == NULL) || (len == 0)) {
        return BASE_TIMER_ERRNO_PARAM_INVALID;
    }

    return attrSetFunc[type](&attrParam->param, value, len);
}

BaseTimerHandle BASE_TimerCreate(TimerAppInfo *appInfo, BaseTimerUsrCb *usrCb, uint32_t timeoutMs, BaseTimerAttr attr)
{
    uint32_t ret;
    BaseTimerInfo *tmrInfo = NULL;
    BaseTimerParamAttr *attrParam = (BaseTimerParamAttr *)attr;

    do {
        if ((appInfo == NULL) || (usrCb == NULL) || (usrCb->proc == NULL) || (timeoutMs == 0) ||
            ((attrParam != NULL) && (attrParam->magic != BASE_TIMER_ATTR_MAGIC))) {
            LOG_INNER_ERR("Timer create param invalid.");
            break;
        }

        tmrInfo = BaseTimerCreateInfo(appInfo, usrCb, timeoutMs, attrParam);
        if (tmrInfo == NULL) {
            LOG_INNER_ERR("Timer create node course mistake.");
            break;
        }

        ret = BaseTimerSetSch(tmrInfo);
        if (ret != 0) {
            LOG_INNER_ERR("Timer add to sch course mistake. (ret = %#x)", ret);
            BaseTimerDestroyInfo(tmrInfo);
            tmrInfo = NULL;
            break;
        }
    } while (0);

    BASE_TimerDestroyAttr(appInfo, attr);

    return tmrInfo;
}

#define BASE_TIMER_DEFAULT_TIME_INTERVEL (10)
void BASE_TimerDestroy(BaseTimerHandle handle)
{
    uint32_t ret;
    BaseTimerInfo *tmrInfo = handle;

    if (!BaseTimerHandleValid(tmrInfo)) {
        LOG_INNER_ERR("Timer destroy handle invalid.");
        return;
    }

    ret = BaseTimerUpdateTime(tmrInfo, BASE_TIMER_DEFAULT_TIME_INTERVEL); /* 异步删除 */
    if (ret != BASE_TIMER_ERRNO_OK) {
        return;
    }

    tmrInfo->delTag = BASE_TIMER_DEL_TAG_TRUE;

    return;
}

uint32_t BASE_TimerStart(BaseTimerHandle handle)
{
    uint32_t ret;
    BaseTimerInfo *tmrInfo = handle;

    if (!BaseTimerHandleValid(tmrInfo)) {
        return BASE_TIMER_ERRNO_HANDLE_INVALID;
    }

    ret = BaseTimerUpdateTime(tmrInfo, tmrInfo->timeoutMs);
    if (ret != BASE_TIMER_ERRNO_OK) {
        return ret;
    }

    return BASE_TIMER_ERRNO_OK;
}

uint32_t BASE_TimerStop(BaseTimerHandle handle)
{
    uint32_t ret;
    BaseTimerInfo *tmrInfo = handle;

    if (!BaseTimerHandleValid(tmrInfo)) {
        return BASE_TIMER_ERRNO_HANDLE_INVALID;
    }

    ret = BaseTimerUpdateTime(tmrInfo, 0);
    if (ret != BASE_TIMER_ERRNO_OK) {
        return ret;
    }

    return BASE_TIMER_ERRNO_OK;
}

uint32_t BASE_TimerResize(BaseTimerHandle handle, uint32_t timeoutMs)
{
    uint32_t ret;
    BaseTimerInfo *tmrInfo = handle;

    if (timeoutMs == 0) {
        return BASE_TIMER_ERRNO_PARAM_INVALID;
    }

    if (!BaseTimerHandleValid(tmrInfo)) {
        return BASE_TIMER_ERRNO_HANDLE_INVALID;
    }

    /* 单次定时器不支持重调超时时间 */
    if ((tmrInfo->optParam != NULL) && (tmrInfo->optParam->mode == BASE_TIMER_MODE_ONE_SHOT)) {
        return BASE_TIMER_ERRNO_NOT_SUPPORT;
    }

    ret = BaseTimerUpdateTime(tmrInfo, timeoutMs);
    if (ret != BASE_TIMER_ERRNO_OK) {
        return ret;
    }

    tmrInfo->timeoutMs = timeoutMs;

    return BASE_TIMER_ERRNO_OK;
}

uint32_t BASE_TimerGetRemainTime(BaseTimerHandle handle, uint32_t *remainMs)
{
    struct itimerspec currValue;
    BaseTimerInfo *tmrInfo = handle;

    if (remainMs == NULL) {
        return BASE_TIMER_ERRNO_PARAM_INVALID;
    }

    if (!BaseTimerHandleValid(tmrInfo)) {
        return BASE_TIMER_ERRNO_HANDLE_INVALID;
    }

    if (timerfd_gettime(tmrInfo->tmrFd, &currValue) != 0) {
        return BASE_TIMER_ERRNO_TIMER_GET_FAILED;
    }

    /* 纳秒除以1000000 加 秒乘以1000 得到毫秒 */
    *remainMs = (uint32_t)(currValue.it_value.tv_nsec / 1000000 + currValue.it_value.tv_sec * 1000);

    return BASE_TIMER_ERRNO_OK;
}

void BASE_TimerCopyAppInfo(TimerAppInfo *dest, TimerAppInfo *source)
{
    dest->appHandle = source->appHandle;
    dest->appCid = source->appCid;
    dest->memHandle = source->memHandle;
    dest->freeCB = source->freeCB;
    dest->mallocCB = source->mallocCB;
    dest->epollCtlCB = source->epollCtlCB;
    return;
}

#ifdef __cplusplus
}
#endif
