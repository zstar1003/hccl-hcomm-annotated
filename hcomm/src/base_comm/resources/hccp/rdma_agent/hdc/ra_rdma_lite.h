/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RA_RDMA_LITE_H
#define RA_RDMA_LITE_H

#include "user_log.h"
#include "rdma_lite.h"
#include "hccp_common.h"

#define DL_ATTRI_VISI_DEF __attribute__((visibility("default")))

#define DL_API_RET_IS_NULL_CHECK(p, str, release_lock)                                                                 \
    do {                                                                                                               \
        if ((p) == NULL) {                                                                                             \
            pthread_mutex_unlock(&release_lock);                                                                       \
            hccp_err("ptr is NULL!, [%s]", (str));                                                                     \
            return (-EINVAL);                                                                                          \
        }                                                                                                              \
    } while (0)

#define DL_API_PTR_IS_NULL_CHECK(p, str)                                                                               \
    do {                                                                                                               \
        if ((p) == NULL) {                                                                                             \
            hccp_warn("%s is NULL!", (str));                                                                           \
        }                                                                                                              \
    } while (0)

struct RaRdmaLiteOps {
    struct rdma_lite_context *(*raRdmaLiteAllocCtx)(u8 phyId, struct dev_cap_info *cap);
    void (*raRdmaLiteFreeCtx)(struct rdma_lite_context *liteCtx);
    int (*raRdmaLiteInitMemPool)(struct rdma_lite_context *liteCtx, struct rdma_lite_mem_attr *liteMemAttr);
    int (*raRdmaLiteDeinitMemPool)(struct rdma_lite_context *liteCtx, u32 memIdx);
    struct rdma_lite_cq *(*raRdmaLiteCreateCq)(struct rdma_lite_context *liteCtx, struct rdma_lite_cq_attr *liteCqAttr);
    int (*raRdmaLiteDestroyCq)(struct rdma_lite_cq *liteCq);
    int (*raRdmaLitePollCq)(struct rdma_lite_cq *liteCq, int numEntries, struct rdma_lite_wc *liteWc);
    int (*raRdmaLitePollCqV2)(struct rdma_lite_cq *liteCq, int numEntries, struct rdma_lite_wc_v2 *liteWc);
    struct rdma_lite_qp *(*raRdmaLiteCreateQp)(struct rdma_lite_context *liteCtx, struct rdma_lite_qp_attr *liteQpAttr);
    int (*raRdmaLiteDestroyQp)(struct rdma_lite_qp *liteQp);
    int (*raRdmaLitePostSend)(struct rdma_lite_qp *liteQp, struct rdma_lite_send_wr *wr,
        struct rdma_lite_send_wr **badWr, struct rdma_lite_post_send_attr *attr, struct rdma_lite_post_send_resp *resp);
    int (*raRdmaLitePostRecv)(struct rdma_lite_qp *liteQp, struct rdma_lite_recv_wr *wr,
        struct rdma_lite_recv_wr **badWr);
    int (*raRdmaLiteSetQpSl)(struct rdma_lite_qp *liteQp, int sl);
    int (*raRdmaLiteCleanQp)(struct rdma_lite_qp *liteQp);
    int (*raRdmaLiteRestoreSnapshot)(struct rdma_lite_context *liteCtx);
    unsigned int (*raRdmaLiteGetApiVersion)(void);
};

DL_ATTRI_VISI_DEF void RaHdcRdmaLiteApiDeinit(void);
DL_ATTRI_VISI_DEF int RaHdcRdmaLiteApiInit(void);
struct rdma_lite_context *RaRdmaLiteAllocCtx(u8 phyId, struct dev_cap_info *cap);

void RaRdmaLiteFreeCtx(struct rdma_lite_context *liteCtx);

struct rdma_lite_cq *RaRdmaLiteCreateCq(struct rdma_lite_context *liteCtx, struct rdma_lite_cq_attr *liteCqAttr);

int RaRdmaLiteDestroyCq(struct rdma_lite_cq *liteCq);

int RaRdmaLitePollCq(struct rdma_lite_cq *liteCq, int numEntries, struct rdma_lite_wc *liteWc);
int RaRdmaLitePollCqV2(struct rdma_lite_cq *liteCq, int numEntries, struct rdma_lite_wc_v2 *liteWc);

struct rdma_lite_qp *RaRdmaLiteCreateQp(struct rdma_lite_context *liteCtx, struct rdma_lite_qp_attr *liteQpAttr);

int RaRdmaLiteDestroyQp(struct rdma_lite_qp *liteQp);

int RaRdmaLitePostSend(struct rdma_lite_qp *liteQp, struct rdma_lite_send_wr *wr, struct rdma_lite_send_wr **badWr,
    struct rdma_lite_post_send_attr *attr, struct rdma_lite_post_send_resp *resp);

int RaRdmaLitePostRecv(struct rdma_lite_qp *liteQp, struct rdma_lite_recv_wr *wr, struct rdma_lite_recv_wr **badWr);

int RaRdmaLiteSetQpSl(struct rdma_lite_qp *liteQp, unsigned char sl);
int RaRdmaLiteCleanQp(struct rdma_lite_qp *liteQp);
int RaRdmaLiteInitMemPool(struct rdma_lite_context *liteCtx, struct rdma_lite_mem_attr *liteMemAttr);
int RaRdmaLiteDeinitMemPool(struct rdma_lite_context *liteCtx, u32 memIdx);
int RaRdmaLiteRestoreSnapshot(struct rdma_lite_context *liteCtx);
unsigned int RaRdmaLiteGetApiVersion(void);
#endif // RA_RDMA_LITE_H
