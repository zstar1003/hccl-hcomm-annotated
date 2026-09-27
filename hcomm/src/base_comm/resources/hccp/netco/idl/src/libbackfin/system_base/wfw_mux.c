/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <errno.h>
#include "bkf_assert.h"
#include "v_stringlib.h"
#include "securec.h"
#include "wfw_mux.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
#pragma pack(4)

struct tagWfwMux {
    WfwMuxInitArg argInit;
    char name[0];
};

#pragma pack()

/* func */
WfwMux *WfwMuxInit(WfwMuxInitArg *arg)
{
    WfwMux *mux = VOS_NULL;
    BOOL argIsInvalid = VOS_FALSE;
    uint32_t strLen;
    uint32_t len;
    int32_t err;

    argIsInvalid = (arg == VOS_NULL) || (arg->name == VOS_NULL) || (arg->memMng == VOS_NULL) ||
                   (arg->attachFd == VOS_NULL) || (arg->reattachFd == VOS_NULL) || (arg->detachFd == VOS_NULL);
    if (argIsInvalid) {
        BKF_ASSERT(0);
        goto error;
    }

    strLen = VOS_StrLen(arg->name);
    len = sizeof(WfwMux) + strLen + 1;
    mux = BKF_MALLOC(arg->memMng, len);
    if (mux == VOS_NULL) {
        BKF_ASSERT(0);
        goto error;
    }
    (void)memset_s(mux, len, 0, len);
    mux->argInit = *arg;
    err = snprintf_truncated_s(mux->name, strLen + 1, "%s", arg->name);
    if (err < 0) {
        goto error;
    }
    mux->argInit.name = mux->name;

    return mux;

error:

    WfwMuxUninit(mux);
    return VOS_NULL;
}

void WfwMuxUninit(WfwMux *mux)
{
    if ((mux == VOS_NULL) || (mux->argInit.memMng == VOS_NULL)) {
        return;
    }

    BKF_FREE(mux->argInit.memMng, mux);
    return;
}

int WfwMuxAttachFd(WfwMux *mux, int fd, uint32_t interestedEvents, F_WFW_MUX_FD_PROC fdProc, void *fdCookie,
    WfwArgForMux *argForMuxOrNull)
{
    if (mux == VOS_NULL) {
        return 0;
    }

    return mux->argInit.attachFd(mux->argInit.cookie, fd, interestedEvents, fdProc, fdCookie, argForMuxOrNull);
}

int WfwMuxReattachFd(WfwMux *mux, int fd, uint32_t interestedEvents, F_WFW_MUX_FD_PROC fdProc, void *fdCookie,
    WfwArgForMux *argForMuxOrNull)
{
    if (mux == VOS_NULL) {
        return 0;
    }

    return mux->argInit.reattachFd(mux->argInit.cookie, fd, interestedEvents, fdProc, fdCookie, argForMuxOrNull);
}

int WfwMuxDetachFd(WfwMux *mux, int fd)
{
    if (mux == VOS_NULL) {
        return 0;
    }

    return mux->argInit.detachFd(mux->argInit.cookie, fd);
}

void *WfwMuxLoopRun(WfwMux *mux)
{
    if (mux == VOS_NULL) {
        return VOS_NULL;
    }
    if (mux->argInit.loopRunOrNull == VOS_NULL) {
        BKF_ASSERT(0);
        errno = EINVAL;
        return VOS_NULL;
    }

    return mux->argInit.loopRunOrNull(mux->argInit.cookie);
}

void WfwMuxStopLoopRun(WfwMux *mux)
{
    if (mux == VOS_NULL) {
        return;
    }
    if (mux->argInit.stopLoopRunOrNull == VOS_NULL) {
        BKF_ASSERT(0);
        errno = EINVAL;
        return;
    }

    mux->argInit.stopLoopRunOrNull(mux->argInit.cookie);
    return;
}

F_WFW_MUX_FD_PROC WfwMuxGetFdProc(WfwMux *mux, int fd, void **fdCookieOut, uint32_t *interestedEventsOut)
{
    if (mux == VOS_NULL) {
        return VOS_NULL;
    }
    if (mux->argInit.getFdProcOrNull == VOS_NULL) {
        BKF_ASSERT(0);
        errno = EINVAL;
        return VOS_NULL;
    }

    return mux->argInit.getFdProcOrNull(mux->argInit.cookie, fd, fdCookieOut, interestedEventsOut);
}

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif
