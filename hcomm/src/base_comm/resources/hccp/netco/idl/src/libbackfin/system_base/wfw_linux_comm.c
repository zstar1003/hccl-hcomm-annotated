/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <unistd.h>
#include <fcntl.h>
#include <execinfo.h>
#include "wfw_linux_comm.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
int WfwSetFdNonBlock(int fd)
{
    int flag;

    flag = fcntl(fd, F_GETFL);
    if (flag == -1) {
        return -1;
    }
    return fcntl(fd, F_SETFL, (unsigned int)flag | O_NONBLOCK);
}

#define WFW_GET_BACK_TRACE_DEPTH_MAX 128
#define WFW_GET_BACK_TRACE_DEPTH_OUT_MAX 7

char *WfwGetBackTraceStr(uint8_t *buf, int32_t bufLen)
{
    (void)buf;
    (void)bufLen;
#ifdef BKF_DEBUG
    BOOL paramIsInvalid = (buf == VOS_NULL) || (bufLen <= 0);
    if (paramIsInvalid) {
        return "__arg_ng__";
    }

    void *ptrs[WFW_GET_BACK_TRACE_DEPTH_MAX];
    int depthTotal = backtrace(ptrs, BKF_GET_ARY_COUNT(ptrs));
    if (depthTotal <= 0) {
        return "__get_ptr_ng__";
    }
    char **strs = backtrace_symbols(ptrs, depthTotal);
    if (strs == NULL) {
        return "__get_str_ng__";
    }

    int depthOut = BKF_GET_MIN(depthTotal - 1, WFW_GET_BACK_TRACE_DEPTH_OUT_MAX);
    int wTotalLen = 0;
    int wLen = snprintf_truncated_s((char *)buf + wTotalLen, bufLen - wTotalLen,
        "\n=====callStack, depthTotal(%d)/out(%d)=====", depthTotal, depthOut);
    if (wLen < 0) {
        goto error;
    }
    wTotalLen += wLen;
    int i;
    for (i = 1; i <= depthOut; i++) {
        char *tempStr = BkfTrimStrPath(strs[i]);
        wLen = snprintf_truncated_s((char *)buf + wTotalLen, bufLen - wTotalLen, "\n[%d]: %s", i, tempStr);
        if (wLen < 0) {
            goto error;
        }

        wTotalLen += wLen;
    }
    wLen = snprintf_truncated_s((char *)buf + wTotalLen, bufLen - wTotalLen, "\n===end=======");
    if (wLen < 0) {
        goto error;
    }
    wTotalLen += wLen;

    free(strs);
    return (char *)buf;

error:
    free(strs);
#endif
    return "__fmt_str_ng__";
}

int WfwGetFuncCallStackDepth(void)
{
    void *ptrs[WFW_GET_BACK_TRACE_DEPTH_MAX];
    int depthTotal = backtrace(ptrs, BKF_GET_ARY_COUNT(ptrs));
    return (depthTotal - 1);
}

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif
