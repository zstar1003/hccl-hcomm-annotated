/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RDMA_USER_IOCTL_CMDS_H
#define RDMA_USER_IOCTL_CMDS_H

#include <linux/types.h>
#include <linux/ioctl.h>

/* Documentation/ioctl/ioctl-number.txt */
#define RDMA_IOCTL_MAGIC 0x1b
#define RDMA_VERBS_IOCTL _IOWR(RDMA_IOCTL_MAGIC, 1, struct ib_uverbs_ioctl_hdr)

enum {
    /* User input */
    UVERBS_ATTR_F_MANDATORY = 1U << 0,
    /*
     * Valid output bit should be ignored and considered set in
     * mandatory fields. This bit is kernel output.
     */
    UVERBS_ATTR_F_VALID_OUTPUT = 1U << 1,
};

struct ib_uverbs_attr {
    __u16 attr_id; /* command specific type attribute */
    __u16 len;     /* only for pointers */
    __u16 flags;   /* combination of UVERBS_ATTR_F_XXXX */
    union {
        struct {
            __u8 elem_id;
            __u8 reserved;
        } enum_data;
        __u16 reserved;
    } attr_data;
    __aligned_u64 data; /* ptr to command, inline data or idr/fd */
};

struct ib_uverbs_ioctl_hdr {
    __u16 length;
    __u16 object_id;
    __u16 method_id;
    __u16 num_attrs;
    __aligned_u64 reserved1;
    __u32 driver_id;
    __u32 reserved2;
    struct ib_uverbs_attr attrs[0];
};

enum rdma_driver_id {
    RDMA_DRIVER_UNKNOWN,
    RDMA_DRIVER_MLX5,
    RDMA_DRIVER_MLX4,
    RDMA_DRIVER_CXGB3,
    RDMA_DRIVER_CXGB4,
    RDMA_DRIVER_MTHCA,
    RDMA_DRIVER_BNXT_RE,
    RDMA_DRIVER_OCRDMA,
    RDMA_DRIVER_NES,
    RDMA_DRIVER_I40IW,
    RDMA_DRIVER_VMW_PVRDMA,
    RDMA_DRIVER_QEDR,
    RDMA_DRIVER_HNS,
    RDMA_DRIVER_USNIC,
    RDMA_DRIVER_RXE,
    RDMA_DRIVER_HFI1,
    RDMA_DRIVER_QIB,
    RDMA_DRIVER_EFA,
    RDMA_DRIVER_SIW,
};

#endif
