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
#include "hccp_dl.h"
#include "rs.h"
#include "dl_netco_function.h"

void *gNetcoApiHandle = NULL;
#ifndef CA_CONFIG_LLT
struct RsNetcoOps gNetcoOps;
#else
struct RsNetcoOps gNetcoOps = {
    .rsNetcoInit = NetCoInitFactory,
    .rsNetcoDeinit = NetCoDestruct,
    .rsNetcoEventDispatch = NetCoFdEventDispatch,
    .rsNetcoTblAddUpd = NetCoTblAddUpd,
};
#endif

STATIC int RsOpenLibnetCoSo(void)
{
#ifndef CA_CONFIG_LLT
    if (gNetcoApiHandle == NULL) {
        gNetcoApiHandle = HccpDlopen("libnet_co.so", RTLD_NOW);
        if (gNetcoApiHandle != NULL) {
            return 0;
        }
        hccp_err("HccpDlopen[libnet_co.so] failed! errno=%d, dlerror: %s", errno, dlerror());
        return -EINVAL;
    } else {
        hccp_run_info("netco_api dlopen again!");
    }
#endif
    return 0;
}

STATIC void RsCloseNetcoSo(void)
{
#ifndef CA_CONFIG_LLT
    if (gNetcoApiHandle != NULL) {
        (void)HccpDlclose(gNetcoApiHandle);
        gNetcoApiHandle = NULL;
    }
#endif
    return;
}

STATIC int RsNetcoTblApiInit(void)
{
#ifndef CA_CONFIG_LLT
    gNetcoOps.rsNetcoInit = (void *(*)(int, NetCoIpPortArg))HccpDlsym(gNetcoApiHandle, "NetCoInitFactory");
    DL_API_RET_IS_NULL_CHECK(gNetcoOps.rsNetcoInit, "NetCoInitFactory");

    gNetcoOps.rsNetcoDeinit = (void (*)(void *))HccpDlsym(gNetcoApiHandle, "NetCoDestruct");
    DL_API_RET_IS_NULL_CHECK(gNetcoOps.rsNetcoDeinit, "NetCoDestruct");

    gNetcoOps.rsNetcoEventDispatch = (unsigned int (*)(void *, int, unsigned int))HccpDlsym(gNetcoApiHandle,
        "NetCoFdEventDispatch");
    DL_API_RET_IS_NULL_CHECK(gNetcoOps.rsNetcoEventDispatch, "NetCoFdEventDispatch");

    gNetcoOps.rsNetcoTblAddUpd = (int (*)(void *, unsigned int, char *, unsigned int))HccpDlsym(gNetcoApiHandle,
        "NetCoTblAddUpd");
    DL_API_RET_IS_NULL_CHECK(gNetcoOps.rsNetcoTblAddUpd, "NetCoTblAddUpd");
#endif
    return 0;
}

STATIC int RsNetcoApiInit(void)
{
    int ret;
    ret = RsOpenLibnetCoSo();
    CHK_PRT_RETURN(ret != 0,
        hccp_err("HccpDlopen[libnet_co.so] failed! ret=[%d],"
                 "Please check network adapter driver has been installed",
            ret),
        ret);

    ret = RsNetcoTblApiInit();
    if (ret != 0) {
        hccp_err("[rs_netco_tbl_api_init]HccpDlopen failed! ret=[%d]", ret);
        RsCloseNetcoSo();
        return ret;
    }

    return 0;
}

int RsNslbApiInit(void)
{
    int ret;

    ret = RsNetcoApiInit();
    CHK_PRT_RETURN(ret, hccp_err("rs_netco_api_init failed! ret=[%d]", ret), ret);

    return 0;
}

void RsNslbApiDeinit(void)
{
    if (gNetcoApiHandle != NULL) {
        (void)HccpDlclose(gNetcoApiHandle);
        gNetcoApiHandle = NULL;
    }
    return;
}

void *RsNetcoInit(int epollfd, NetCoIpPortArg ipPortArg)
{
    if (gNetcoOps.rsNetcoInit == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_netco_init is null");
        return NULL;
#endif
    }
    return gNetcoOps.rsNetcoInit(epollfd, ipPortArg);
}

void RsNetcoDeinit(void *co)
{
    if (gNetcoOps.rsNetcoDeinit == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_netco_deinit is null");
        return;
#endif
    }
    return gNetcoOps.rsNetcoDeinit(co);
}

unsigned int RsNetcoEventDispatch(void *co, int fd, unsigned int curEvents)
{
    if (gNetcoOps.rsNetcoEventDispatch == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_netco_event_dispatch is null");
        return EINVAL;
#endif
    }
    return gNetcoOps.rsNetcoEventDispatch(co, fd, curEvents);
}

int RsNetcoTblAddUpd(void *netcoHandle, unsigned int type, char *data, unsigned int dataLen)
{
    if (gNetcoOps.rsNetcoTblAddUpd == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_netco_tbl_add_upd is null, type(%u)", type);
        return -EINVAL;
#endif
    }
    return gNetcoOps.rsNetcoTblAddUpd(netcoHandle, type, data, dataLen);
}
