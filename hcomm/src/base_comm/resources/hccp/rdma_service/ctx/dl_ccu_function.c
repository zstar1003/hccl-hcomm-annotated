/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "config_log.h"
#include "hccp_dl.h"
#include "hccp_tlv.h"
#include "dl_ccu_function.h"

void *gCcuApiHandle = NULL;
#ifndef CA_CONFIG_LLT
struct RsCcuOps gCcuOps;
#else
struct RsCcuOps gCcuOps = {
    .rsCcuInit = ccu_init,
    .rsCcuUninit = ccu_uninit,
    .rsCcuCustomChannel = ccu_custom_channel,
    .rsCcuGetCqeBaseAddr = ccu_get_cqe_base_addr,
    .rsCcuGetMemInfo = ccu_get_mem_info,
    .rsCcuTlvRequest = ccu_tlv_request,
};
#endif

STATIC int RsCcuDeviceApiInit(void)
{
#ifndef CA_CONFIG_LLT
    gCcuOps.rsCcuInit = (int (*)(void))HccpDlsym(gCcuApiHandle, "ccu_init");
    DL_API_RET_IS_NULL_CHECK(gCcuOps.rsCcuInit, "ccu_init");

    gCcuOps.rsCcuUninit = (int (*)(void))HccpDlsym(gCcuApiHandle, "ccu_uninit");
    DL_API_RET_IS_NULL_CHECK(gCcuOps.rsCcuUninit, "ccu_uninit");

    gCcuOps.rsCcuCustomChannel = (int (*)(const struct channel_info_in *in,
        struct channel_info_out *out))HccpDlsym(gCcuApiHandle, "ccu_custom_channel");
    DL_API_RET_IS_NULL_CHECK(gCcuOps.rsCcuCustomChannel, "ccu_custom_channel");

    gCcuOps.rsCcuGetCqeBaseAddr = (unsigned long long (*)(unsigned int dieId))HccpDlsym(gCcuApiHandle,
        "ccu_get_cqe_base_addr");
    DL_API_RET_IS_NULL_CHECK(gCcuOps.rsCcuGetCqeBaseAddr, "ccu_get_cqe_base_addr");

    gCcuOps.rsCcuGetMemInfo = (int (*)(unsigned int dieId, unsigned long long memTypeBitmap,
        struct ccu_mem_rsp *rsp))HccpDlsym(gCcuApiHandle, "ccu_get_mem_info");
    DL_API_RET_IS_NULL_CHECK(gCcuOps.rsCcuGetMemInfo, "ccu_get_mem_info");

    gCcuOps.rsCcuTlvRequest = (int (*)(struct ccu_tlv_request_info *tlvRequestInfo))HccpDlsym(gCcuApiHandle,
        "ccu_tlv_request");
    DL_API_RET_IS_NULL_INFO(gCcuOps.rsCcuTlvRequest, "ccu_tlv_request");
#endif
    return 0;
}

STATIC int RsOpenCcuSo(void)
{
#ifndef CA_CONFIG_LLT
    if (gCcuApiHandle == NULL) {
        gCcuApiHandle = HccpDlopen("libccu-user-drv.so", RTLD_NOW);
        if (gCcuApiHandle != NULL) {
            return 0;
        }
        return -EINVAL;
    } else {
        hccp_run_info("ccu_api dlopen again!");
    }
#endif
    return 0;
}

STATIC void RsCloseCcuSo(void)
{
#ifndef CA_CONFIG_LLT
    if (gCcuApiHandle != NULL) {
        (void)HccpDlclose(gCcuApiHandle);
        gCcuApiHandle = NULL;
    }
#endif
    return;
}

int RsCcuApiInit(void)
{
    int ret;

    ret = RsOpenCcuSo();
    CHK_PRT_RETURN(ret,
        hccp_err("HccpDlopen[libccu-user-drv.so] failed! ret=[%d],"
                 "Please check ccu driver has been installed",
            ret),
        ret);

    ret = RsCcuDeviceApiInit();
    if (ret != 0) {
        hccp_err("[rs_ccu_device_api_init]HccpDlopen failed! ret=[%d]", ret);
        RsCloseCcuSo();
        return ret;
    }
    return 0;
}

void RsCcuApiDeinit(void)
{
    RsCloseCcuSo();
    return;
}

int RsCcuInit(void)
{
    if (gCcuApiHandle == NULL || gCcuOps.rsCcuInit == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("g_ccu_api_handle is NULL or rs_ccu_init is NULL");
        return -EINVAL;
#endif
    }
    return gCcuOps.rsCcuInit();
}

int RsCcuCustomChannel(const struct channel_info_in *in, struct channel_info_out *out)
{
    if (gCcuApiHandle == NULL || gCcuOps.rsCcuCustomChannel == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("g_ccu_api_handle is NULL or rs_ccu_custom_channel is NULL");
        return -EINVAL;
#endif
    }
    return gCcuOps.rsCcuCustomChannel(in, out);
}

int RsCcuGetCqeBaseAddr(unsigned int dieId, unsigned long long *cqeBaseAddr)
{
    if (gCcuApiHandle == NULL || gCcuOps.rsCcuGetCqeBaseAddr == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("g_ccu_api_handle is NULL or rs_ccu_get_cqe_base_addr is NULL");
        return -EINVAL;
#endif
    }
    CHK_PRT_RETURN(cqeBaseAddr == NULL, hccp_err("cqe_base_addr is null, dieId:%u", dieId), -EINVAL);
    *cqeBaseAddr = gCcuOps.rsCcuGetCqeBaseAddr(dieId);
    return 0;
}

int RsCcuUninit(void)
{
    if (gCcuApiHandle == NULL || gCcuOps.rsCcuUninit == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("g_ccu_api_handle is NULL or rs_ccu_uninit is NULL");
        return -EINVAL;
#endif
    }
    return gCcuOps.rsCcuUninit();
}

int RsCcuGetMemInfo(char *dataIn, char *dataOut, unsigned int *bufferSize)
{
    struct ccu_mem_rsp *rsp = (struct ccu_mem_rsp *)dataOut;
    struct CcuMemReq *memReq = (struct CcuMemReq *)dataIn;

    if (gCcuApiHandle == NULL || gCcuOps.rsCcuGetMemInfo == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("g_ccu_api_handle is NULL or rsCcuGetMemInfo is NULL");
        return -EINVAL;
#endif
    }
    *bufferSize = sizeof(struct ccu_mem_rsp);
    return gCcuOps.rsCcuGetMemInfo(memReq->udieIdx, memReq->memTypeBitmap, rsp);
}

bool isCcuTlvReqExist(void)
{
    if (gCcuApiHandle == NULL || gCcuOps.rsCcuTlvRequest == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_info_rma("g_ccu_api_handle is NULL or rsCcuTlvRequest is NULL");
        return false;
#endif
    }
    return true;
}

int RsCcuTlvRequest(unsigned int type, char *dataIn, char *dataOut, unsigned int dataInlength,
    unsigned int *dataOutLength, unsigned int dataOutMaxLength)
{
    int ret = 0;
    struct ccu_tlv_request_info tlvRequestInfo = {
        .ccuMsg = type,
        .dataIn = dataIn,
        .dataInLength = dataInlength,
        .dataOut = dataOut,
        .dataOutMaxLength = dataOutMaxLength,
    };
    ret = gCcuOps.rsCcuTlvRequest(&tlvRequestInfo);
    *dataOutLength = tlvRequestInfo.dataOutLength;
    return ret;
}