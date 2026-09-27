/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <dlfcn.h>
#include <pthread.h>
#include "ra.h"
#include "ra_rs_err.h"
#include "ra_rdma_lite.h"

static pthread_mutex_t gRdmaLiteApiLock = PTHREAD_MUTEX_INITIALIZER;
void *gRdmaLiteApiHandle = NULL;
int gRdmaLiteApiRefcnt = 0;
#ifndef HNS_ROCE_LLT
struct RaRdmaLiteOps gRdmaLiteOps;
#else
struct RaRdmaLiteOps gRdmaLiteOps = {
    .raRdmaLiteAllocCtx = rdma_lite_alloc_context,
    .raRdmaLiteFreeCtx = rdma_lite_free_context,
    .raRdmaLiteInitMemPool = rdma_lite_init_mem_pool,
    .raRdmaLiteDeinitMemPool = rdma_lite_deinit_mem_pool,
    .raRdmaLiteCreateCq = rdma_lite_create_cq,
    .raRdmaLiteDestroyCq = rdma_lite_destroy_cq,
    .raRdmaLitePollCq = rdma_lite_poll_cq,
    .raRdmaLitePollCqV2 = rdma_lite_poll_cq_v2,
    .raRdmaLiteCreateQp = rdma_lite_create_qp,
    .raRdmaLiteDestroyQp = rdma_lite_destroy_qp,
    .raRdmaLitePostSend = rdma_lite_post_send,
    .raRdmaLitePostRecv = rdma_lite_post_recv,
    .raRdmaLiteSetQpSl = rdma_lite_set_qp_sl,
    .raRdmaLiteCleanQp = rdma_lite_clean_qp,
    .raRdmaLiteRestoreSnapshot = rdma_lite_restore_snapshot,
};
#endif

STATIC int RaHdcOpenRdmaLiteSo(void)
{
#ifndef HNS_ROCE_LLT
    if (gRdmaLiteApiHandle == NULL) {
        gRdmaLiteApiHandle = dlopen("libascend_rdma_lite.so", RTLD_NOW);
        if (gRdmaLiteApiHandle != NULL) {
            return 0;
        }
        return -EINVAL;
    } else {
        hccp_run_info("rdma lite api dlopen again!");
    }
#endif
    return 0;
}

#ifndef HNS_ROCE_LLT
static int RaRdmaLiteControlPlaneApiInit(void)
{
    gRdmaLiteOps.raRdmaLiteAllocCtx = (struct rdma_lite_context * (*)(u8 phyId, struct dev_cap_info * cap))
        dlsym(gRdmaLiteApiHandle, "rdma_lite_alloc_context");
    DL_API_RET_IS_NULL_CHECK(gRdmaLiteOps.raRdmaLiteAllocCtx, "rdma_lite_alloc_context", gRdmaLiteApiLock);

    gRdmaLiteOps.raRdmaLiteFreeCtx = (void (*)(struct rdma_lite_context *liteCtx))dlsym(gRdmaLiteApiHandle,
        "rdma_lite_free_context");
    DL_API_RET_IS_NULL_CHECK(gRdmaLiteOps.raRdmaLiteFreeCtx, "rdma_lite_free_context", gRdmaLiteApiLock);

    gRdmaLiteOps.raRdmaLiteInitMemPool = (int (*)(struct rdma_lite_context *liteCtx,
        struct rdma_lite_mem_attr *liteMemAttr))dlsym(gRdmaLiteApiHandle, "rdma_lite_init_mem_pool");
    DL_API_PTR_IS_NULL_CHECK(gRdmaLiteOps.raRdmaLiteInitMemPool, "rdma_lite_init_mem_pool");

    gRdmaLiteOps.raRdmaLiteDeinitMemPool = (int (*)(struct rdma_lite_context *liteCtx,
        u32 memIdx))dlsym(gRdmaLiteApiHandle, "rdma_lite_deinit_mem_pool");
    DL_API_PTR_IS_NULL_CHECK(gRdmaLiteOps.raRdmaLiteDeinitMemPool, "rdma_lite_deinit_mem_pool");

    gRdmaLiteOps.raRdmaLiteCreateCq = (struct rdma_lite_cq *
                                       (*)(struct rdma_lite_context * liteCtx, struct rdma_lite_cq_attr * liteCqAttr))
        dlsym(gRdmaLiteApiHandle, "rdma_lite_create_cq");
    DL_API_RET_IS_NULL_CHECK(gRdmaLiteOps.raRdmaLiteCreateCq, "rdma_lite_create_cq", gRdmaLiteApiLock);

    gRdmaLiteOps.raRdmaLiteDestroyCq = (int (*)(struct rdma_lite_cq *liteCq))dlsym(gRdmaLiteApiHandle,
        "rdma_lite_destroy_cq");
    DL_API_RET_IS_NULL_CHECK(gRdmaLiteOps.raRdmaLiteDestroyCq, "rdma_lite_destroy_cq", gRdmaLiteApiLock);

    gRdmaLiteOps.raRdmaLiteCreateQp = (struct rdma_lite_qp *
                                       (*)(struct rdma_lite_context * liteCtx, struct rdma_lite_qp_attr * liteQpAttr))
        dlsym(gRdmaLiteApiHandle, "rdma_lite_create_qp");
    DL_API_RET_IS_NULL_CHECK(gRdmaLiteOps.raRdmaLiteCreateQp, "rdma_lite_create_qp", gRdmaLiteApiLock);

    gRdmaLiteOps.raRdmaLiteDestroyQp = (int (*)(struct rdma_lite_qp *liteQp))dlsym(gRdmaLiteApiHandle,
        "rdma_lite_destroy_qp");
    DL_API_RET_IS_NULL_CHECK(gRdmaLiteOps.raRdmaLiteDestroyQp, "rdma_lite_destroy_qp", gRdmaLiteApiLock);

    gRdmaLiteOps.raRdmaLiteSetQpSl = (int (*)(struct rdma_lite_qp *liteQp, int sl))dlsym(gRdmaLiteApiHandle,
        "rdma_lite_set_qp_sl");
    DL_API_RET_IS_NULL_CHECK(gRdmaLiteOps.raRdmaLiteSetQpSl, "rdma_lite_set_qp_sl", gRdmaLiteApiLock);

    gRdmaLiteOps.raRdmaLiteCleanQp = (int (*)(struct rdma_lite_qp *liteQp))dlsym(gRdmaLiteApiHandle,
        "rdma_lite_clean_qp");
    DL_API_PTR_IS_NULL_CHECK(gRdmaLiteOps.raRdmaLiteCleanQp, "rdma_lite_clean_qp");

    gRdmaLiteOps.raRdmaLiteRestoreSnapshot = (int (*)(struct rdma_lite_context *liteCtx))dlsym(gRdmaLiteApiHandle,
        "rdma_lite_restore_snapshot");
    DL_API_PTR_IS_NULL_CHECK(gRdmaLiteOps.raRdmaLiteRestoreSnapshot, "rdma_lite_restore_snapshot");

    gRdmaLiteOps.raRdmaLiteGetApiVersion = (unsigned int (*)(void))dlsym(gRdmaLiteApiHandle,
        "rdma_lite_get_api_version");
    DL_API_PTR_IS_NULL_CHECK(gRdmaLiteOps.raRdmaLiteGetApiVersion, "rdma_lite_get_api_version");

    return 0;
}

static int RaRdmaLiteDataPlaneApiInit(void)
{
    gRdmaLiteOps.raRdmaLitePostSend = (int (*)(struct rdma_lite_qp *liteQp, struct rdma_lite_send_wr *wr,
        struct rdma_lite_send_wr **badWr, struct rdma_lite_post_send_attr *attr,
        struct rdma_lite_post_send_resp *resp))dlsym(gRdmaLiteApiHandle, "rdma_lite_post_send");
    DL_API_RET_IS_NULL_CHECK(gRdmaLiteOps.raRdmaLitePostSend, "rdma_lite_post_send", gRdmaLiteApiLock);

    gRdmaLiteOps.raRdmaLitePostRecv = (int (*)(struct rdma_lite_qp *liteQp, struct rdma_lite_recv_wr *wr,
        struct rdma_lite_recv_wr **badWr))dlsym(gRdmaLiteApiHandle, "rdma_lite_post_recv");
    DL_API_PTR_IS_NULL_CHECK(gRdmaLiteOps.raRdmaLitePostRecv, "rdma_lite_post_recv");

    gRdmaLiteOps.raRdmaLitePollCq = (int (*)(struct rdma_lite_cq *liteCq, int numEntries,
        struct rdma_lite_wc *liteWc))dlsym(gRdmaLiteApiHandle, "rdma_lite_poll_cq");
    DL_API_RET_IS_NULL_CHECK(gRdmaLiteOps.raRdmaLitePollCq, "rdma_lite_poll_cq", gRdmaLiteApiLock);

    gRdmaLiteOps.raRdmaLitePollCqV2 = (int (*)(struct rdma_lite_cq *liteCq, int numEntries,
        struct rdma_lite_wc_v2 *liteWc))dlsym(gRdmaLiteApiHandle, "rdma_lite_poll_cq_v2");
    DL_API_PTR_IS_NULL_CHECK(gRdmaLiteOps.raRdmaLitePollCqV2, "rdma_lite_poll_cq_v2");
    return 0;
}
#endif

DL_ATTRI_VISI_DEF int RaHdcRdmaLiteApiInit(void)
{
#ifndef HNS_ROCE_LLT
    int ret;

    pthread_mutex_lock(&gRdmaLiteApiLock);
    if (gRdmaLiteApiHandle != NULL) {
        gRdmaLiteApiRefcnt++;
        pthread_mutex_unlock(&gRdmaLiteApiLock);
        return 0;
    }

    ret = RaHdcOpenRdmaLiteSo();
    if (ret) {
        pthread_mutex_unlock(&gRdmaLiteApiLock);
        hccp_err("HccpDlopen[libascend_rdma_lite.so] "
                 "failed! ret=[%d][%s]. Please check rdma lite driver has been installed.",
            ret, dlerror());
        return ret;
    }

    ret = RaRdmaLiteControlPlaneApiInit();
    if (ret != 0) {
        return ret;
    }

    ret = RaRdmaLiteDataPlaneApiInit();
    if (ret != 0) {
        return ret;
    }

    gRdmaLiteApiRefcnt++;
    pthread_mutex_unlock(&gRdmaLiteApiLock);
#endif
    return 0;
}

DL_ATTRI_VISI_DEF void RaHdcRdmaLiteApiDeinit(void)
{
    pthread_mutex_lock(&gRdmaLiteApiLock);
    if (gRdmaLiteApiHandle != NULL) {
        gRdmaLiteApiRefcnt--;
        if (gRdmaLiteApiRefcnt > 0) {
            pthread_mutex_unlock(&gRdmaLiteApiLock);
            return;
        }
        (void)dlclose(gRdmaLiteApiHandle);
        gRdmaLiteApiHandle = NULL;
    }
    pthread_mutex_unlock(&gRdmaLiteApiLock);

    return;
}

struct rdma_lite_context *RaRdmaLiteAllocCtx(u8 phyId, struct dev_cap_info *cap)
{
    if (gRdmaLiteApiHandle == NULL || gRdmaLiteOps.raRdmaLiteAllocCtx == NULL) {
#ifndef HNS_ROCE_LLT
        hccp_err("ra_rdma_lite_alloc_ctx is null");
        return NULL;
#endif
    }
    return gRdmaLiteOps.raRdmaLiteAllocCtx(phyId, cap);
}

void RaRdmaLiteFreeCtx(struct rdma_lite_context *liteCtx)
{
    if (gRdmaLiteApiHandle == NULL || gRdmaLiteOps.raRdmaLiteFreeCtx == NULL) {
#ifndef HNS_ROCE_LLT
        hccp_err("ra_rdma_lite_free_ctx is null");
        return;
#endif
    }
    return gRdmaLiteOps.raRdmaLiteFreeCtx(liteCtx);
}

struct rdma_lite_cq *RaRdmaLiteCreateCq(struct rdma_lite_context *liteCtx, struct rdma_lite_cq_attr *liteCqAttr)
{
    if (gRdmaLiteApiHandle == NULL || gRdmaLiteOps.raRdmaLiteCreateCq == NULL) {
#ifndef HNS_ROCE_LLT
        hccp_err("ra_rdma_lite_create_cq is null");
        return NULL;
#endif
    }
    return gRdmaLiteOps.raRdmaLiteCreateCq(liteCtx, liteCqAttr);
}

int RaRdmaLiteDestroyCq(struct rdma_lite_cq *liteCq)
{
    if (gRdmaLiteApiHandle == NULL || gRdmaLiteOps.raRdmaLiteDestroyCq == NULL) {
#ifndef HNS_ROCE_LLT
        hccp_err("ra_rdma_lite_destroy_cq is null");
        return -EINVAL;
#endif
    }
    return gRdmaLiteOps.raRdmaLiteDestroyCq(liteCq);
}

int RaRdmaLitePollCq(struct rdma_lite_cq *liteCq, int numEntries, struct rdma_lite_wc *liteWc)
{
    if (gRdmaLiteApiHandle == NULL || gRdmaLiteOps.raRdmaLitePollCq == NULL) {
#ifndef HNS_ROCE_LLT
        hccp_err("ra_rdma_lite_poll_cq is null");
        return -EINVAL;
#endif
    }
    return gRdmaLiteOps.raRdmaLitePollCq(liteCq, numEntries, liteWc);
}

int RaRdmaLitePollCqV2(struct rdma_lite_cq *liteCq, int numEntries, struct rdma_lite_wc_v2 *liteWc)
{
    if (gRdmaLiteApiHandle == NULL || gRdmaLiteOps.raRdmaLitePollCqV2 == NULL) {
#ifndef HNS_ROCE_LLT
        hccp_err("ra_rdma_lite_poll_cq_v2 is null");
        return -EINVAL;
#endif
    }
    return gRdmaLiteOps.raRdmaLitePollCqV2(liteCq, numEntries, liteWc);
}

struct rdma_lite_qp *RaRdmaLiteCreateQp(struct rdma_lite_context *liteCtx, struct rdma_lite_qp_attr *liteQpAttr)
{
    if (gRdmaLiteApiHandle == NULL || gRdmaLiteOps.raRdmaLiteCreateQp == NULL) {
#ifndef HNS_ROCE_LLT
        hccp_err("ra_rdma_lite_create_qp is null");
        return NULL;
#endif
    }
    return gRdmaLiteOps.raRdmaLiteCreateQp(liteCtx, liteQpAttr);
}

int RaRdmaLiteDestroyQp(struct rdma_lite_qp *liteQp)
{
    if (gRdmaLiteApiHandle == NULL || gRdmaLiteOps.raRdmaLiteDestroyQp == NULL) {
#ifndef HNS_ROCE_LLT
        hccp_err("ra_rdma_lite_destroy_qp is null");
        return -EINVAL;
#endif
    }
    return gRdmaLiteOps.raRdmaLiteDestroyQp(liteQp);
}

int RaRdmaLitePostSend(struct rdma_lite_qp *liteQp, struct rdma_lite_send_wr *wr, struct rdma_lite_send_wr **badWr,
    struct rdma_lite_post_send_attr *attr, struct rdma_lite_post_send_resp *resp)
{
    if (gRdmaLiteApiHandle == NULL || gRdmaLiteOps.raRdmaLitePostSend == NULL) {
#ifndef HNS_ROCE_LLT
        hccp_err("ra_rdma_lite_post_send is null");
        return -EINVAL;
#endif
    }
    return gRdmaLiteOps.raRdmaLitePostSend(liteQp, wr, badWr, attr, resp);
}

int RaRdmaLitePostRecv(struct rdma_lite_qp *liteQp, struct rdma_lite_recv_wr *wr, struct rdma_lite_recv_wr **badWr)
{
    if (gRdmaLiteApiHandle == NULL || gRdmaLiteOps.raRdmaLitePostRecv == NULL) {
#ifndef HNS_ROCE_LLT
        hccp_err("ra_rdma_lite_post_recv is null");
        return -EINVAL;
#endif
    }
    return gRdmaLiteOps.raRdmaLitePostRecv(liteQp, wr, badWr);
}

int RaRdmaLiteSetQpSl(struct rdma_lite_qp *liteQp, unsigned char sl)
{
    if (gRdmaLiteApiHandle == NULL || gRdmaLiteOps.raRdmaLiteSetQpSl == NULL) {
#ifndef HNS_ROCE_LLT
        hccp_err("ra_rdma_lite_set_qp_sl is null");
        return -EINVAL;
#endif
    }
    return gRdmaLiteOps.raRdmaLiteSetQpSl(liteQp, sl);
}

int RaRdmaLiteCleanQp(struct rdma_lite_qp *liteQp)
{
    if (gRdmaLiteApiHandle == NULL || gRdmaLiteOps.raRdmaLiteCleanQp == NULL) {
#ifndef HNS_ROCE_LLT
        hccp_err("ra_rdma_lite_clean_qp is null");
        return -EINVAL;
#endif
    }
    return gRdmaLiteOps.raRdmaLiteCleanQp(liteQp);
}

int RaRdmaLiteInitMemPool(struct rdma_lite_context *liteCtx, struct rdma_lite_mem_attr *liteMemAttr)
{
    if (gRdmaLiteApiHandle == NULL || gRdmaLiteOps.raRdmaLiteInitMemPool == NULL) {
#ifndef HNS_ROCE_LLT
        hccp_err("ra_rdma_lite_init_mem_pool is null");
        return -EINVAL;
#endif
    }
    return gRdmaLiteOps.raRdmaLiteInitMemPool(liteCtx, liteMemAttr);
}

int RaRdmaLiteDeinitMemPool(struct rdma_lite_context *liteCtx, u32 memIdx)
{
    if (gRdmaLiteApiHandle == NULL || gRdmaLiteOps.raRdmaLiteDeinitMemPool == NULL) {
#ifndef HNS_ROCE_LLT
        hccp_err("ra_rdma_lite_deinit_mem_pool is null");
        return -EINVAL;
#endif
    }
    return gRdmaLiteOps.raRdmaLiteDeinitMemPool(liteCtx, memIdx);
}

int RaRdmaLiteRestoreSnapshot(struct rdma_lite_context *liteCtx)
{
    if (gRdmaLiteApiHandle == NULL || gRdmaLiteOps.raRdmaLiteRestoreSnapshot == NULL) {
#ifndef HNS_ROCE_LLT
        hccp_err("driver package may not support ra_rdma_lite_restore_snapshot interface, please change new one");
        return -ENOTSUPP;
#endif
    }
    return gRdmaLiteOps.raRdmaLiteRestoreSnapshot(liteCtx);
}

unsigned int RaRdmaLiteGetApiVersion(void)
{
    if (gRdmaLiteApiHandle != NULL && gRdmaLiteOps.raRdmaLiteGetApiVersion != NULL) {
        return gRdmaLiteOps.raRdmaLiteGetApiVersion();
    }

    return 0;
}
