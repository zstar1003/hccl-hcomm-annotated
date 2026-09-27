/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccp_dl.h"
#include "net_adapt_u_api.h"
#include "dl_net_function.h"

void *gNetApiHandle = NULL;
#ifndef CA_CONFIG_LLT
struct RsNetOps gNetOps;
#else
struct RsNetOps gNetOps = {
    .rsNetAdaptInit = net_adapt_init,
    .rsNetAdaptUninit = net_adapt_uninit,
    .rsNetAllocJfcId = net_alloc_jfc_id,
    .rsNetFreeJfcId = net_free_jfc_id,
    .rsNetAllocJettyId = net_alloc_jetty_id,
    .rsNetFreeJettyId = net_free_jetty_id,
    .rsNetGetCqeBaseAddr = net_get_cqe_base_addr,
    .rsNetGetApiVersion = net_get_api_version,
};
#endif

int RsNetAdaptApiInit(void)
{
#ifndef CA_CONFIG_LLT
    gNetOps.rsNetAdaptInit = (int (*)(void))HccpDlsym(gNetApiHandle, "net_adapt_init");
    DL_API_RET_IS_NULL_CHECK(gNetOps.rsNetAdaptInit, "net_adapt_init");

    gNetOps.rsNetAdaptUninit = (void (*)(void))HccpDlsym(gNetApiHandle, "net_adapt_uninit");
    DL_API_RET_IS_NULL_CHECK(gNetOps.rsNetAdaptUninit, "net_adapt_uninit");

    gNetOps.rsNetAllocJfcId = (int (*)(const char *udevName, unsigned int jfcMode,
        unsigned int *jfcId))HccpDlsym(gNetApiHandle, "net_alloc_jfc_id");
    DL_API_RET_IS_NULL_CHECK(gNetOps.rsNetAllocJfcId, "net_alloc_jfc_id");

    gNetOps.rsNetFreeJfcId = (int (*)(const char *udevName, unsigned int jfcMode,
        unsigned int jfcId))HccpDlsym(gNetApiHandle, "net_free_jfc_id");
    DL_API_RET_IS_NULL_CHECK(gNetOps.rsNetFreeJfcId, "net_free_jfc_id");

    gNetOps.rsNetAllocJettyId = (int (*)(const char *udevName, unsigned int jettyMode,
        unsigned int *jettyId))HccpDlsym(gNetApiHandle, "net_alloc_jetty_id");
    DL_API_RET_IS_NULL_CHECK(gNetOps.rsNetAllocJettyId, "net_alloc_jetty_id");

    gNetOps.rsNetFreeJettyId = (int (*)(const char *udevName, unsigned int jettyMode,
        unsigned int jettyId))HccpDlsym(gNetApiHandle, "net_free_jetty_id");
    DL_API_RET_IS_NULL_CHECK(gNetOps.rsNetFreeJettyId, "net_free_jetty_id");

    gNetOps.rsNetGetCqeBaseAddr = (unsigned long long (*)(unsigned int dieId))HccpDlsym(gNetApiHandle,
        "net_get_cqe_base_addr");
    DL_API_RET_IS_NULL_CHECK(gNetOps.rsNetGetCqeBaseAddr, "net_get_cqe_base_addr");

    gNetOps.rsNetGetApiVersion = (unsigned int (*)(void))HccpDlsym(gNetApiHandle, "net_get_api_version");
#endif
    return 0;
}

int RsOpenNetSo(void)
{
#ifndef CA_CONFIG_LLT
    if (gNetApiHandle == NULL) {
        gNetApiHandle = HccpDlopen("libnet_adapt.so", RTLD_NOW);
        return ((gNetApiHandle != NULL) ? 0 : -EINVAL);
    }
    hccp_run_info("net_adapt_api HccpDlopen again!");
#endif
    return 0;
}

void RsCloseNetSo(void)
{
#ifndef CA_CONFIG_LLT
    if (gNetApiHandle != NULL) {
        (void)HccpDlclose(gNetApiHandle);
        gNetApiHandle = NULL;
    }
#endif
    return;
}

int RsNetApiInit(void)
{
    int ret;

    ret = RsOpenNetSo();
    CHK_PRT_RETURN(ret != 0,
        hccp_err("rs_open_net_so[libnet_adapt.so] failed! ret=[%d],"
                 "please check network adapter driver has been installed",
            ret),
        ret);

    ret = RsNetAdaptApiInit();
    if (ret != 0) {
        hccp_err("rs_net_adapt_api_init failed! ret=[%d]", ret);
        RsCloseNetSo();
        return ret;
    }
    return 0;
}

void RsNetApiDeinit(void)
{
    RsCloseNetSo();
    return;
}

int RsNetAdaptInit(void)
{
    if (gNetApiHandle == NULL || gNetOps.rsNetAdaptInit == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("g_net_api_handle is NULL or rs_net_adapt_init is NULL");
        return -EINVAL;
#endif
    }
    return gNetOps.rsNetAdaptInit();
}

void RsNetAdaptUninit(void)
{
    if (gNetApiHandle == NULL || gNetOps.rsNetAdaptUninit == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("g_net_api_handle is NULL or rs_net_adapt_uninit is NULL");
        return;
#endif
    }
    gNetOps.rsNetAdaptUninit();
}

int RsNetAllocJfcId(const char *udevName, unsigned int jfcMode, unsigned int *jfcId)
{
    if (gNetApiHandle == NULL || gNetOps.rsNetAllocJfcId == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("g_net_api_handle is NULL or rs_net_alloc_jfc_id is NULL");
        return -EINVAL;
#endif
    }
    return gNetOps.rsNetAllocJfcId(udevName, jfcMode, jfcId);
}

int RsNetFreeJfcId(const char *udevName, unsigned int jfcMode, unsigned int jfcId)
{
    if (gNetApiHandle == NULL || gNetOps.rsNetFreeJfcId == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("g_net_api_handle is NULL or rs_net_free_jfc_id is NULL");
        return -EINVAL;
#endif
    }
    return gNetOps.rsNetFreeJfcId(udevName, jfcMode, jfcId);
}

int RsNetAllocJettyId(const char *udevName, unsigned int jettyMode, unsigned int *jettyId)
{
    if (gNetApiHandle == NULL || gNetOps.rsNetAllocJettyId == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("g_net_api_handle is NULL or rs_net_alloc_jetty_id is NULL");
        return -EINVAL;
#endif
    }
    return gNetOps.rsNetAllocJettyId(udevName, jettyMode, jettyId);
}

int RsNetFreeJettyId(const char *udevName, unsigned int jettyMode, unsigned int jettyId)
{
    if (gNetApiHandle == NULL || gNetOps.rsNetFreeJettyId == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("g_net_api_handle is NULL or rs_net_free_jetty_id is NULL");
        return -EINVAL;
#endif
    }
    return gNetOps.rsNetFreeJettyId(udevName, jettyMode, jettyId);
}

int RsNetGetCqeBaseAddr(unsigned int dieId, unsigned long long *cqeBaseAddr)
{
    if (gNetApiHandle == NULL || gNetOps.rsNetGetCqeBaseAddr == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("g_net_api_handle is NULL or rs_net_get_cqe_base_addr is NULL");
        return -EINVAL;
#endif
    }
    CHK_PRT_RETURN(cqeBaseAddr == NULL, hccp_err("cqe_base_addr is null, dieId:%u", dieId), -EINVAL);
    *cqeBaseAddr = gNetOps.rsNetGetCqeBaseAddr(dieId);
    return 0;
}

unsigned int RsNetGetApiVersion(void)
{
    if (gNetApiHandle == NULL || gNetOps.rsNetGetApiVersion == NULL) {
        return 0;
    }

    return gNetOps.rsNetGetApiVersion();
}
