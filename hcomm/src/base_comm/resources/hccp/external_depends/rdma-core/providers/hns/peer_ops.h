/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PEER_OPS_H
#define PEER_OPS_H
#include <stddef.h>
#include <stdint.h>
#include <infiniband/verbs.h>

// gdr_develop
#define IBV_EXP_PEER_IOMEMORY ((struct ibv_exp_peer_buf *)-1UL)

struct ibv_exp_peer_buf {
    void *addr;
    size_t length;
    /* Reserved for future extensions, must be 0 */
    uint32_t comp_mask;
};

struct ibv_exp_peer_direct_attr {
    uint64_t peer_id; // peer id，代表每个GPU的编号
    uint64_t (*register_va)(void *start, size_t length, uint64_t peer_id,
        struct ibv_exp_peer_buf *pb); // 将va地址注册到device，返回一个device ptr
};

enum ibv_exp_peer_op {
    IBV_EXP_PEER_OP_DB,
};

struct peer_op_wr {
    struct peer_op_wr *next;   // next peer wr
    enum ibv_exp_peer_op type; // peer wr type
    union {
        struct {
            uint64_t data;      // target_id value
            uint64_t target_id; // target_id addr
        } qw;

        struct {
            uint64_t db_addr; // doorbell addr
            uint64_t db_val;  // doorbell value
        } db;
    } wr;
};

struct ibv_exp_peer_commit {
    struct peer_op_wr *storage; // peer wr addr
    uint32_t entries;           // peer wr num
};
#endif
