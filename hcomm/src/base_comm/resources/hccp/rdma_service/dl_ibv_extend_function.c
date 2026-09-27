/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "config_log.h"
#include <pthread.h>
#include "hccp_dl.h"
#include "dl_ibv_extend_function.h"

void *gRoceIbvExtendApiHandle = NULL;
#ifndef CA_CONFIG_LLT
STATIC struct RsIbvExtendOps gIbvExtendOps = {0};
#else
struct RsIbvExtendOps gIbvExtendOps = {
    .rsIbvExtendGetVersion = ibv_extend_get_version,
    .rsIbvExtendCheckVersion = ibv_extend_check_version,
    .rsIbvOpenExtend = ibv_open_extend,
    .rsIbvCloseExtend = ibv_close_extend,
    .rsIbvQueryDeviceExtend = ibv_query_device_extend,
    .rsIbvCreateQpExtend = ibv_create_qp_extend,
    .rsIbvCreateCqExtend = ibv_create_cq_extend,
    .rsIbvDestroyQpExtend = ibv_destroy_qp_extend,
    .rsIbvDestroyCqExtend = ibv_destroy_cq_extend,
    .rsIbvModifyQpExtend = ibv_modify_qp_extend,
    .rsIbvQueryQpSupportedHyroceFeature = ibv_query_qp_supported_hyroce_feature,
    .rsIbvNegoQpHyroceFeature = ibv_nego_qp_hyroce_feature,
};
#endif

STATIC int RsIbvExtendIbvApiInit(void)
{
#ifndef CA_CONFIG_LLT
    gIbvExtendOps.rsIbvExtendGetVersion = (const char *(*)(uint32_t *, uint32_t *,
        uint32_t *))HccpDlsym(gRoceIbvExtendApiHandle, "ibv_extend_get_version");
    DL_API_RET_IS_NULL_CHECK(gIbvExtendOps.rsIbvExtendGetVersion, "ibv_extend_get_version");
    gIbvExtendOps.rsIbvExtendCheckVersion = (int (*)(uint32_t, uint32_t, uint32_t))HccpDlsym(gRoceIbvExtendApiHandle,
        "ibv_extend_check_version");
    DL_API_RET_IS_NULL_CHECK(gIbvExtendOps.rsIbvExtendCheckVersion, "ibv_extend_check_version");
    gIbvExtendOps.rsIbvOpenExtend = (struct ibv_context_extend * (*)(struct ibv_context *))
        HccpDlsym(gRoceIbvExtendApiHandle, "ibv_open_extend");
    DL_API_RET_IS_NULL_CHECK(gIbvExtendOps.rsIbvOpenExtend, "ibv_open_extend");
    gIbvExtendOps.rsIbvCloseExtend = (int (*)(struct ibv_context_extend *))HccpDlsym(gRoceIbvExtendApiHandle,
        "ibv_close_extend");
    DL_API_RET_IS_NULL_CHECK(gIbvExtendOps.rsIbvCloseExtend, "ibv_close_extend");
    gIbvExtendOps.rsIbvQueryDeviceExtend = (int (*)(struct ibv_context_extend *,
        struct ibv_device_attr_extend *))HccpDlsym(gRoceIbvExtendApiHandle, "ibv_query_device_extend");
    DL_API_RET_IS_NULL_CHECK(gIbvExtendOps.rsIbvQueryDeviceExtend, "ibv_query_device_extend");
    gIbvExtendOps.rsIbvCreateQpExtend = (struct ibv_qp_extend *
                                         (*)(struct ibv_context_extend *, struct ibv_qp_init_attr_extend *))
        HccpDlsym(gRoceIbvExtendApiHandle, "ibv_create_qp_extend");
    DL_API_RET_IS_NULL_CHECK(gIbvExtendOps.rsIbvCreateQpExtend, "ibv_create_qp_extend");
    gIbvExtendOps.rsIbvCreateCqExtend = (struct ibv_cq_extend *
                                         (*)(struct ibv_context_extend *, struct ibv_cq_init_attr_extend *))
        HccpDlsym(gRoceIbvExtendApiHandle, "ibv_create_cq_extend");
    DL_API_RET_IS_NULL_CHECK(gIbvExtendOps.rsIbvCreateCqExtend, "ibv_create_cq_extend");
    gIbvExtendOps.rsIbvDestroyQpExtend = (int (*)(struct ibv_context_extend *,
        struct ibv_qp_extend *))HccpDlsym(gRoceIbvExtendApiHandle, "ibv_destroy_qp_extend");
    DL_API_RET_IS_NULL_CHECK(gIbvExtendOps.rsIbvDestroyQpExtend, "ibv_destroy_qp_extend");
    gIbvExtendOps.rsIbvDestroyCqExtend = (int (*)(struct ibv_context_extend *,
        struct ibv_cq_extend *))HccpDlsym(gRoceIbvExtendApiHandle, "ibv_destroy_cq_extend");
    DL_API_RET_IS_NULL_CHECK(gIbvExtendOps.rsIbvDestroyCqExtend, "ibv_destroy_cq_extend");
    gIbvExtendOps.rsIbvModifyQpExtend = (int (*)(struct ibv_context_extend *, struct ibv_qp_attr_extend *,
        int))HccpDlsym(gRoceIbvExtendApiHandle, "ibv_modify_qp_extend");
    DL_API_RET_IS_NULL_INFO(gIbvExtendOps.rsIbvModifyQpExtend, "ibv_modify_qp_extend");
    gIbvExtendOps.rsIbvQueryQpSupportedHyroceFeature = (int (*)(struct ibv_context_extend *, struct ibv_qp *, uint32_t,
        uint32_t, struct ibv_hyroce_feature *))HccpDlsym(gRoceIbvExtendApiHandle,
        "ibv_query_qp_supported_hyroce_feature");
    DL_API_RET_IS_NULL_INFO(gIbvExtendOps.rsIbvQueryQpSupportedHyroceFeature, "ibv_query_qp_supported_hyroce_feature");
    gIbvExtendOps.rsIbvNegoQpHyroceFeature = (int (*)(struct ibv_context_extend *, struct ibv_qp *,
        const struct ibv_hyroce_feature *, struct ibv_hyroce_feature *, uint32_t *))HccpDlsym(gRoceIbvExtendApiHandle,
        "ibv_nego_qp_hyroce_feature");
    DL_API_RET_IS_NULL_INFO(gIbvExtendOps.rsIbvNegoQpHyroceFeature, "ibv_nego_qp_hyroce_feature");
#endif
    return 0;
}

STATIC int RsOpenIbvExtendSo(void)
{
#ifndef CA_CONFIG_LLT
    if (gRoceIbvExtendApiHandle == NULL) {
        gRoceIbvExtendApiHandle = HccpDlopen("libibv_extend.so", RTLD_NOW | RTLD_GLOBAL);
        if (gRoceIbvExtendApiHandle != NULL) {
            return 0;
        }
        return -EINVAL;
    } else {
        hccp_run_info("IbvExtendApi dlopen again!");
    }
#endif
    return 0;
}

STATIC void RsCloseIbvExtendSo(void)
{
#ifndef CA_CONFIG_LLT
    if (gRoceIbvExtendApiHandle != NULL) {
        (void)HccpDlclose(gRoceIbvExtendApiHandle);
        gRoceIbvExtendApiHandle = NULL;
    }
#endif
    return;
}

int RsIbvExtendApiInit(void)
{
#ifndef CA_CONFIG_LLT
    int ret = 0;

    ret = RsOpenIbvExtendSo();
    if (ret != 0) {
        hccp_warn_others("HccpDlopen[libibv_extend.so] doesn't exist!");
        return 0;
    }

    ret = RsIbvExtendIbvApiInit();
    if (ret != 0) {
        hccp_err("RsIbvExtendIbvApiInit failed! ret:%d", ret);
        RsCloseIbvExtendSo();
        return ret;
    }
#endif
    return 0;
}

void RsIbvExtendApiDeinit(void)
{
    RsCloseIbvExtendSo();
    return;
}

const char *RsIbvExtendGetVersion(uint32_t *major, uint32_t *minor, uint32_t *patch)
{
    if (gIbvExtendOps.rsIbvExtendGetVersion == NULL) {
#ifndef CA_CONFIG_LLT
        return NULL;
#endif
    }
    return gIbvExtendOps.rsIbvExtendGetVersion(major, minor, patch);
}

int RsIbvExtendCheckVersion(uint32_t driverMajor, uint32_t driverMinor, uint32_t driverPatch)
{
    if (gIbvExtendOps.rsIbvExtendCheckVersion == NULL) {
#ifndef CA_CONFIG_LLT
        return -EINVAL;
#endif
    }
    return gIbvExtendOps.rsIbvExtendCheckVersion(driverMajor, driverMinor, driverPatch);
}

struct ibv_context_extend *RsIbvOpenExtend(struct ibv_context *context)
{
    if (gIbvExtendOps.rsIbvOpenExtend == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_warn_others("rsIbvOpenExtend is null");
        return NULL;
#endif
    }
    return gIbvExtendOps.rsIbvOpenExtend(context);
}

int RsIbvCloseExtend(struct ibv_context_extend *context)
{
    if (context == NULL) {
        return 0;
    }

    if (gIbvExtendOps.rsIbvCloseExtend == NULL) {
#ifndef CA_CONFIG_LLT
        return 0;
#endif
    }
    return gIbvExtendOps.rsIbvCloseExtend(context);
}

int RsIbvQueryDeviceExtend(struct ibv_context_extend *context, struct ibv_device_attr_extend *extDevAttr)
{
    if (gIbvExtendOps.rsIbvQueryDeviceExtend == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rsIbvQueryDeviceExtend is null");
        return -EINVAL;
#endif
    }
    return gIbvExtendOps.rsIbvQueryDeviceExtend(context, extDevAttr);
}

struct ibv_cq_extend *RsIbvCreateCqExtend(struct ibv_context_extend *context,
    struct ibv_cq_init_attr_extend *cqInitAttr)
{
    if (gIbvExtendOps.rsIbvCreateCqExtend == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rsIbvCreateCqExtend is null");
        return NULL;
#endif
    }
    return gIbvExtendOps.rsIbvCreateCqExtend(context, cqInitAttr);
}

int RsIbvDestroyCqExtend(struct ibv_context_extend *context, void *ibvCqExt)
{
    struct ibv_cq_extend *cqExtend = (struct ibv_cq_extend *)ibvCqExt;

    if (gIbvExtendOps.rsIbvDestroyCqExtend == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rsIbvDestroyCqExtend is null");
        return -EINVAL;
#endif
    }
    return gIbvExtendOps.rsIbvDestroyCqExtend(context, cqExtend);
}

struct ibv_qp_extend *RsIbvCreateQpExtend(struct ibv_context_extend *context,
    struct ibv_qp_init_attr_extend *qpInitAttr)
{
    if (gIbvExtendOps.rsIbvCreateQpExtend == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rsIbvCreateQpExtend is null");
        return NULL;
#endif
    }
    return gIbvExtendOps.rsIbvCreateQpExtend(context, qpInitAttr);
}

int RsIbvDestroyQpExtend(struct ibv_context_extend *context, struct ibv_qp_extend *qpExtend)
{
    if (gIbvExtendOps.rsIbvDestroyQpExtend == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rsIbvDestroyQpExtend is null");
        return -EINVAL;
#endif
    }
    return gIbvExtendOps.rsIbvDestroyQpExtend(context, qpExtend);
}

int RsIbvModifyQpExtend(struct ibv_context_extend *context, struct ibv_qp_attr_extend *attr, int attr_mask)
{
    if (gIbvExtendOps.rsIbvModifyQpExtend == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_warn_others("rsIbvModifyQpExtend is null");
        return -EINVAL;
#endif
    }
    return gIbvExtendOps.rsIbvModifyQpExtend(context, attr, attr_mask);
}

int RsIbvQueryQpSupportedHyroceFeature(struct ibv_context_extend *context, struct ibv_qp *qp, uint32_t sl, uint32_t tc,
    struct ibv_hyroce_feature *feature)
{
    if (gIbvExtendOps.rsIbvQueryQpSupportedHyroceFeature == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_warn_others("rsIbvQueryQpSupportedHyroceFeature is null");
        return -EINVAL;
#endif
    }
    return gIbvExtendOps.rsIbvQueryQpSupportedHyroceFeature(context, qp, sl, tc, feature);
}

int RsIbvNegoQpHyroceFeature(struct ibv_context_extend *context, struct ibv_qp *qp,
    const struct ibv_hyroce_feature *input, struct ibv_hyroce_feature *output, uint32_t *needMoreNego)
{
    if (gIbvExtendOps.rsIbvNegoQpHyroceFeature == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_warn_others("rsIbvNegoQpHyroceFeature is null");
        return -EINVAL;
#endif
    }
    return gIbvExtendOps.rsIbvNegoQpHyroceFeature(context, qp, input, output, needMoreNego);
}
