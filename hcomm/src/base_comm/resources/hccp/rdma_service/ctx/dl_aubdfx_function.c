/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccp_dl.h"
#include "aubdfx_api.h"
#include "ra_rs_err.h"
#include "urma_types.h"
#include "dl_aubdfx_function.h"

STATIC void *gAubdfxApiHandle = NULL;
STATIC struct RsAubdfxOps gAubdfxOps;

STATIC int RsAubdfxNotifyApiInit(void)
{
    gAubdfxOps.rsAubdfxNotifyEvent = (int (*)(unsigned int, unsigned int, void *,
        unsigned int))HccpDlsym(gAubdfxApiHandle, "aubdfx_notify_event");
    DL_API_RET_IS_NULL_CHECK(gAubdfxOps.rsAubdfxNotifyEvent, "aubdfx_notify_event");
    return 0;
}

STATIC int RsOpenAubdfxSo(void)
{
    if (gAubdfxApiHandle != NULL) {
        hccp_run_info("aubdfxApi HccpDlopen again!");
        return 0;
    }

    gAubdfxApiHandle = HccpDlopen("libaubdfx_u.so", RTLD_NOW);
    if (gAubdfxApiHandle != NULL) {
        hccp_info("[notify][event]dlopen libaubdfx_u.so success");
        return 0;
    }
    return -EINVAL;
}

STATIC void RsCloseAubdfxSo(void)
{
    if (gAubdfxApiHandle != NULL) {
        (void)HccpDlclose(gAubdfxApiHandle);
        gAubdfxApiHandle = NULL;
    }
}

int RsAubdfxApiInit(void)
{
    int ret = RsOpenAubdfxSo();

    CHK_PRT_RETURN(ret != 0, hccp_warn("rsOpenAubdfxSo[libaubdfx_u.so] unsuccessful! ret=[%d]", ret), ret);

    ret = RsAubdfxNotifyApiInit();
    if (ret != 0) {
        hccp_warn("rsAubdfxNotifyApiInit unsuccessful! ret=[%d]", ret);
        RsCloseAubdfxSo();
        return ret;
    }
    return 0;
}

void RsAubdfxApiDeinit(void)
{
    RsCloseAubdfxSo();
}

STATIC void RsAubdfxPrintUbServiceErrinfo(struct ub_service_errinfo *errInfo)
{
    hccp_info("[notify][event]ubServiceErrinfo: dieid:%u ueid:%u servicetype:%u errortype:%u", errInfo->dieid,
        errInfo->ueid, errInfo->servicetype, errInfo->errortype);
    hccp_info("[notify][event]srceid:" EID_FMT " dsteid:" EID_FMT, EID_RAW_ARGS(((uint8_t *)errInfo->srceid)),
        EID_RAW_ARGS(((uint8_t *)errInfo->dsteid)));
    hccp_info("[notify][event]value:0x%llx", (unsigned long long)errInfo->value);
}

int RsAubdfxNotifyEvent(unsigned int dieId, unsigned int notifyCmd, void *data, unsigned int dataLen)
{
    if (gAubdfxApiHandle == NULL || gAubdfxOps.rsAubdfxNotifyEvent == NULL) {
        hccp_err("gAubdfxApiHandle is NULL or rsAubdfxNotifyEvent is NULL");
        return -ENOTSUPP;
    }

    if (notifyCmd == 0) {
        RsAubdfxPrintUbServiceErrinfo((struct ub_service_errinfo *)data);
    }
    return gAubdfxOps.rsAubdfxNotifyEvent(dieId, notifyCmd, data, dataLen);
}
