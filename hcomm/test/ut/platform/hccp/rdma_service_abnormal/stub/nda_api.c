/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdlib.h>
#include "ibv_extend.h"

const char* ibv_extend_get_version(uint32_t* major, uint32_t* minor, uint32_t* patch)
{
    if (major != NULL) {
        *major = IBV_EXTEND_VERSION_MAJOR;
    }
    if (minor != NULL) {
        *minor = IBV_EXTEND_VERSION_MINOR;
    }
    if (patch != NULL) {
        *patch = IBV_EXTEND_VERSION_PATCH;
    }
    return IBV_EXTEND_VERSION_STRING;
}

int ibv_extend_check_version(uint32_t driver_major, uint32_t driver_minor, uint32_t driver_patch) { return 0; }

struct ibv_context_extend* ibv_open_extend(struct ibv_context* context)
{
    struct ibv_context_extend* ctx_extend = NULL;

    ctx_extend = malloc(sizeof(struct ibv_context_extend));
    if (NULL == ctx_extend) {
        return NULL;
    }
    memset(ctx_extend, 0, sizeof(struct ibv_context_extend));

    return ctx_extend;
}

int ibv_close_extend(struct ibv_context_extend* context)
{
    if (context != NULL) {
        free(context);
    }

    return 0;
}

int ibv_query_device_extend(struct ibv_context_extend* context, struct ibv_device_attr_extend* ext_dev_attr)
{
    return 0;
}

struct ibv_qp_extend*
ibv_create_qp_extend(struct ibv_context_extend* context, struct ibv_qp_init_attr_extend* qp_init_attr)
{
    struct ibv_qp_extend* qp_extend = NULL;

    qp_extend = malloc(sizeof(struct ibv_qp_extend));
    if (NULL == qp_extend) {
        return NULL;
    }
    memset(qp_extend, 0, sizeof(struct ibv_qp_extend));

    return qp_extend;
}

struct ibv_cq_extend*
ibv_create_cq_extend(struct ibv_context_extend* context, struct ibv_cq_init_attr_extend* cq_init_attr)
{
    struct ibv_cq_extend* cq_extend = NULL;

    cq_extend = malloc(sizeof(struct ibv_cq_extend));
    if (NULL == cq_extend) {
        return NULL;
    }
    memset(cq_extend, 0, sizeof(struct ibv_cq_extend));

    return cq_extend;
}

struct ibv_srq_extend*
ibv_create_srq_extend(struct ibv_context_extend* context, struct ibv_srq_init_attr_extend* srq_init_attr)
{
    struct ibv_srq_extend* srq_extend = NULL;

    srq_extend = malloc(sizeof(struct ibv_srq_extend));
    if (NULL == srq_extend) {
        return NULL;
    }
    memset(srq_extend, 0, sizeof(struct ibv_srq_extend));

    return srq_extend;
}

int ibv_destroy_qp_extend(struct ibv_context_extend* context, struct ibv_qp_extend* qp_extend)
{
    if (qp_extend != NULL) {
        free(qp_extend);
    }

    return 0;
}

int ibv_destroy_cq_extend(struct ibv_context_extend* context, struct ibv_cq_extend* cq_extend)
{
    if (cq_extend != NULL) {
        free(cq_extend);
    }

    return 0;
}

int ibv_destroy_srq_extend(struct ibv_context_extend* context, struct ibv_srq_extend* srq_extend)
{
    if (srq_extend != NULL) {
        free(srq_extend);
    }

    return 0;
}

void verbs_register_driver_extend(const struct verbs_device_extend_ops* ops) { return; }

int ibv_modify_qp_extend(struct ibv_context_extend* context, struct ibv_qp_attr_extend* attr, int attr_mask)
{
    return 0;
}

int ibv_query_qp_supported_hyroce_feature(
    struct ibv_context_extend* context, struct ibv_qp* qp, uint32_t sl, uint32_t tc, struct ibv_hyroce_feature* feature)
{
    return 0;
}

int ibv_nego_qp_hyroce_feature(
    struct ibv_context_extend* context, struct ibv_qp* qp, const struct ibv_hyroce_feature* input,
    struct ibv_hyroce_feature* output, uint32_t* needMoreNego)
{
    return 0;
}
