/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _HNS_ROCE_LITE_STDIO_H
#define _HNS_ROCE_LITE_STDIO_H

#include "rdma_lite.h"
#include "hns_roce_lite.h"

static inline u64 align_down(u64 val, u64 align)
{
    return (val) & ~(align - 1);
}

static inline u64 align_up(u64 val, u32 align)
{
    return (val + align - 1) & ~(align - 1);
}

int hns_roce_verify_lite_qp(struct rdma_lite_qp_attr *attr);
void hns_roce_set_lite_qp_attr(struct hns_roce_lite_qp *qp, struct rdma_lite_qp_attr *attr);
int hns_roce_lite_qp_lock_init(struct hns_roce_lite_qp *qp);
void hns_roce_init_lite_qp_indices(struct hns_roce_lite_qp *qp);
void hns_roce_lite_qp_lock_uninit(struct hns_roce_lite_qp *qp);
int hns_roce_store_lite_qp(struct hns_roce_lite_context *ctx, uint32_t qpn, struct hns_roce_lite_qp *qp);
void hns_roce_clear_lite_qp(struct hns_roce_lite_context *ctx, uint32_t qpn);
void *hns_roce_lite_mmap_host_va(u64 device_va, u32 device_va_len, struct hns_roce_lite_context *ctx);
int hns_roce_lite_unmmap_host_va(u64 device_va, struct hns_roce_lite_context *ctx);
int hns_roce_lite_mmap_hva(struct rdma_lite_device_buf *dev_buf, struct rdma_lite_host_buf *host_buf,
    struct hns_roce_lite_context *ctx);
int hns_roce_lite_unmmap_hva(struct rdma_lite_host_buf *host_buf, struct hns_roce_lite_context *ctx);
int hns_roce_lite_mmap_hdb(struct rdma_lite_device_buf *dev_buf, struct rdma_lite_host_buf *host_buf,
    struct hns_roce_lite_context *ctx);
int hns_roce_lite_unmmap_hdb(struct rdma_lite_host_buf *host_buf, struct hns_roce_lite_context *ctx);
#endif /* _HNS_ROCE_LITE_STDIO_H */