/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RDMA_LITE_H
#define RDMA_LITE_H

#include "rdma_lite_common.h"

/**
 * API VERSION NUMBER combines major version, minor version and patch version, version range form 0x00 to 0xff.
 * example : 0x020103 means version 0x020103, major 0x02, minor 0x01, patch 0x03
 * when delete API, modify API name, should add major version.
 * when add new API, should add minor version.
 * when modify enum para, struct para add patch version. this means when new API compatible with old API
 */
#define LITE_API_VER_MAJOR 0x0
#define LITE_API_VER_MINOR 0x0
#define LITE_API_VER_PATCH 0x1
#define LITE_API_VERSION ((LITE_API_VER_MAJOR << 16U) | (LITE_API_VER_MINOR << 8U) | (LITE_API_VER_PATCH))
#define LITE_WC_EXT_VERSION 1

typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short int u16;
typedef unsigned char u8;

struct rdma_lite_host_buf {
    u64 dva;    // Device侧的VA
    void *hva;  // host侧的VA
    u32 length; // 长度
    u32 mem_idx;
};

struct rdma_lite_mem_pool {
    struct rdma_lite_host_buf host_buf;
    u32 offset;
    u32 used_size;
};

struct rdma_lite_context {
    struct dev_cap_info cap; // 设备能力
};

struct rdma_lite_mem_attr {
    struct rdma_lite_device_mem_attr mem_data;
};

struct rdma_lite_cq_attr {
    struct rdma_lite_device_cq_attr device_cq_attr;
    u32 mem_idx;
};

struct rdma_lite_cq {
    struct rdma_lite_context *ctx;
};

enum rdma_lite_qp_state {
    RDMA_LITE_QPS_RESET,
    RDMA_LITE_QPS_INIT,
    RDMA_LITE_QPS_RTR,
    RDMA_LITE_QPS_RTS,
    RDMA_LITE_QPS_SQD,
    RDMA_LITE_QPS_SQE,
    RDMA_LITE_QPS_ERR,
    RDMA_LITE_QPS_UNKNOWN
};

enum rdma_lite_qp_type {
    RDMA_LITE_QPT_RC = 2,
    RDMA_LITE_QPT_UC,
    RDMA_LITE_QPT_UD,
    RDMA_LITE_QPT_RAW_PACKET = 8,
    RDMA_LITE_QPT_XRC_SEND = 9,
    RDMA_LITE_QPT_XRC_RECV,
    RDMA_LITE_QPT_DRIVER = 0xff
};

struct rdma_lite_qp_cap {
    u32 max_send_wr;
    u32 max_recv_wr;
    u32 max_send_sge;
    u32 max_recv_sge;
    u32 max_inline_data;
};

struct rdma_lite_qp_attr {
    struct rdma_lite_device_qp_attr device_qp_attr;
    struct rdma_lite_cq *send_cq;     // QP SQ关联的CQ
    struct rdma_lite_cq *recv_cq;     // QP RQ关联的CQ
    enum rdma_lite_qp_type qp_type;   // QP的服务类型
    enum rdma_lite_qp_state qp_state; // QP的状态位
    int sq_sig_all;                   // QP SQ的WR是否需要产生CQE标记，1：产生，0：不产生
    struct rdma_lite_qp_cap cap;      // QP SQ&RQ的能力
    int qp_mode;                      // 指示哪种类型的QP，定义如下：
                                      // NOR_QP_MODE       = 0
                                      // GDR_TMPL_QP_MODE  = 1
                                      // OP_QP_MODE        = 2
                                      // GDR_ASYN_QP_MODE  = 3
                                      // OP_QP_MODE_EXT    = 4
    u32 mem_idx;
};

struct rdma_lite_qp {
    struct rdma_lite_context *ctx;
    struct rdma_lite_cq *send_cq;
    struct rdma_lite_cq *recv_cq;
    enum rdma_lite_qp_type qp_type;
    enum rdma_lite_qp_state qp_state;
    u32 qp_num;
};

enum rdma_lite_wc_status {
    RDMA_LITE_WC_SUCCESS,
    RDMA_LITE_WC_LOC_LEN_ERR,
    RDMA_LITE_WC_LOC_QP_OP_ERR,
    RDMA_LITE_WC_LOC_EEC_OP_ERR,
    RDMA_LITE_WC_LOC_PROT_ERR,
    RDMA_LITE_WC_WR_FLUSH_ERR,
    RDMA_LITE_WC_MW_BIND_ERR,
    RDMA_LITE_WC_BAD_RESP_ERR,
    RDMA_LITE_WC_LOC_ACCESS_ERR,
    RDMA_LITE_WC_REM_INV_REQ_ERR,
    RDMA_LITE_WC_REM_ACCESS_ERR,
    RDMA_LITE_WC_REM_OP_ERR,
    RDMA_LITE_WC_RETRY_EXC_ERR,
    RDMA_LITE_WC_RNR_RETRY_EXC_ERR,
    RDMA_LITE_WC_LOC_RDD_VIOL_ERR,
    RDMA_LITE_WC_REM_INV_RD_REQ_ERR,
    RDMA_LITE_WC_REM_ABORT_ERR,
    RDMA_LITE_WC_INV_EECN_ERR,
    RDMA_LITE_WC_INV_EEC_STATE_ERR,
    RDMA_LITE_WC_FATAL_ERR,
    RDMA_LITE_WC_RESP_TIMEOUT_ERR,
    RDMA_LITE_WC_GENERAL_ERR,
    RDMA_LITE_WC_TM_ERR,
    RDMA_LITE_WC_TM_RNDV_INCOMPLETE
};

enum rdma_lite_wc_opcode {
    RDMA_LITE_WC_SEND,
    RDMA_LITE_WC_RDMA_WRITE,
    RDMA_LITE_WC_RDMA_READ,
    RDMA_LITE_WC_COMP_SWAP,
    RDMA_LITE_WC_FETCH_ADD,
    RDMA_LITE_WC_BIND_MW,
    RDMA_LITE_WC_LOCAL_INV,
    RDMA_LITE_WC_TSO,
    RDMA_LITE_WC_RECV = (1 << 7),
    RDMA_LITE_WC_RECV_RDMA_WITH_IMM,

    RDMA_LITE_WC_TM_ADD,
    RDMA_LITE_WC_TM_DEL,
    RDMA_LITE_WC_TM_SYNC,
    RDMA_LITE_WC_TM_RECV,
    RDMA_LITE_WC_TM_NO_TAG,
    RDMA_LITE_WC_WRITE_WITH_NOTIFY = 0xf2,
    RDMA_LITE_WC_REDUCE_WRITE = 0xf5,
    RDMA_LITE_WC_REDUCE_WRITE_WITH_NOTIFY = 0xf6,
    RDMA_LITE_WC_DRIVER1
};

enum rdma_lite_wc_flags {
    RDMA_LITE_WC_GRH = (1 << 0),
    RDMA_LITE_WC_WITH_IMM = (1 << 1),
    RDMA_LITE_WC_IP_CSUM_OK = (1 << 2),
    RDMA_LITE_WC_WITH_INV = (1 << 3),
    RDMA_LITE_WC_TM_SYNC_REQ = (1 << 4),
    RDMA_LITE_WC_TM_MATCH = (1 << 5),
    RDMA_LITE_WC_TM_DATA_VALID = (1 << 6)
};

struct rdma_lite_wc {
    u64 wr_id;
    enum rdma_lite_wc_status status;
    enum rdma_lite_wc_opcode opcode;
    u32 vendor_err;
    u32 byte_len;
    u32 qp_num;
    u32 wc_flags; // 参见enum rdma_lite_wc_flags的定义
};

struct rdma_lite_wc_ext {
    union {
        u32 imm_data;
        u32 invalidated_rkey;
    };
    u16 resv[5U];
    u32 version;
};

struct rdma_lite_wc_v2 {
    struct rdma_lite_wc wc;
    struct rdma_lite_wc_ext ext;
};

struct rdma_lite_sge {
    u64 addr;
    u32 length;
    u32 lkey;
};

enum rdma_lite_wr_opcode {
    RDMA_LITE_WR_RDMA_WRITE,
    RDMA_LITE_WR_RDMA_WRITE_WITH_IMM,
    RDMA_LITE_WR_SEND,
    RDMA_LITE_WR_SEND_WITH_IMM,
    RDMA_LITE_WR_RDMA_READ,
    RDMA_LITE_WR_ATOMIC_CMP_AND_SWP,
    RDMA_LITE_WR_ATOMIC_FETCH_AND_ADD,
    RDMA_LITE_WR_LOCAL_INV,
    RDMA_LITE_WR_BIND_MW,
    RDMA_LITE_WR_SEND_WITH_INV,
    RDMA_LITE_WR_TSO,
    RDMA_LITE_WR_ATOMIC_WRITE = 0xf0,
    RDMA_LITE_WR_WRITE_WITH_NOTIFY = 0xf2,
    RDMA_LITE_WR_NOP = 0xf3,
    RDMA_LITE_WR_REDUCE_WRITE = 0xf5,
    RDMA_LITE_WR_REDUCE_WRITE_NOTIFY = 0xf6,
    RDMA_LITE_WR_DRIVER1,
};

enum rdma_lite_send_flags {
    RDMA_LITE_SEND_FENCE = (1 << 0),
    RDMA_LITE_SEND_SIGNALED = (1 << 1),
    RDMA_LITE_SEND_SOLICITED = (1 << 2),
    RDMA_LITE_SEND_INLINE = (1 << 3),
    RDMA_LITE_SEND_IP_CSUM = (1 << 4)
};

struct rdma_lite_send_wr {
    u64 wr_id;
    struct rdma_lite_send_wr *next;
    struct rdma_lite_sge *sg_list;
    int num_sge;
    enum rdma_lite_wr_opcode opcode;
    u32 send_flags; // 参见enum rdma_lite_send_flags的定义
    u64 remote_addr;
    u32 rkey;
    u32 imm_data;
};

struct rdma_lite_post_send_attr {
    u8 reduce_op;
    u8 reduce_type;
};

struct lite_wqe_info {
    u32 sq_index;  // index of SQ
    u32 wqe_index; // index of WQE
};

struct lite_db_info {
    u32 db_index;               // index of DB
    unsigned long lite_db_info; // DB content
};

struct rdma_lite_post_send_resp {
    union {
        struct lite_wqe_info wqe_tmp; // wqe template info
        struct lite_db_info db;       // doorbell info
    };
};

struct rdma_lite_recv_wr {
    u64 wr_id;
    struct rdma_lite_recv_wr *next;
    struct rdma_lite_sge *sg_list;
    int num_sge;
};

struct rdma_lite_context *rdma_lite_alloc_context(u8 phy_id, struct dev_cap_info *cap);
void rdma_lite_free_context(struct rdma_lite_context *lite_ctx);

int rdma_lite_init_mem_pool(struct rdma_lite_context *lite_ctx, struct rdma_lite_mem_attr *lite_mem_attr);
int rdma_lite_deinit_mem_pool(struct rdma_lite_context *lite_ctx, u32 mem_idx);

struct rdma_lite_cq *rdma_lite_create_cq(struct rdma_lite_context *lite_ctx, struct rdma_lite_cq_attr *lite_cq_attr);
int rdma_lite_destroy_cq(struct rdma_lite_cq *lite_cq);

struct rdma_lite_qp *rdma_lite_create_qp(struct rdma_lite_context *lite_ctx, struct rdma_lite_qp_attr *lite_qp_attr);
int rdma_lite_destroy_qp(struct rdma_lite_qp *lite_qp);

int rdma_lite_poll_cq(struct rdma_lite_cq *lite_cq, int num_entries, struct rdma_lite_wc *lite_wc);
int rdma_lite_poll_cq_v2(struct rdma_lite_cq *lite_cq, int num_entries, struct rdma_lite_wc_v2 *lite_wc);

int rdma_lite_post_send(struct rdma_lite_qp *lite_qp, struct rdma_lite_send_wr *wr, struct rdma_lite_send_wr **bad_wr,
    struct rdma_lite_post_send_attr *attr, struct rdma_lite_post_send_resp *resp);

int rdma_lite_post_recv(struct rdma_lite_qp *lite_qp, struct rdma_lite_recv_wr *wr, struct rdma_lite_recv_wr **bad_wr);

int rdma_lite_set_qp_sl(struct rdma_lite_qp *lite_qp, int sl);

int rdma_lite_clean_qp(struct rdma_lite_qp *lite_qp);

int rdma_lite_restore_snapshot(struct rdma_lite_context *lite_ctx);

unsigned int rdma_lite_get_api_version(void);

struct rdma_lite_ops {
    struct rdma_lite_context *(*rdma_lite_alloc_context)(u8 phy_id, struct dev_cap_info *cap);
    void (*rdma_lite_free_context)(struct rdma_lite_context *lite_ctx);

    int (*rdma_lite_init_mem_pool)(struct rdma_lite_context *lite_ctx, struct rdma_lite_mem_attr *lite_mem_attr);
    int (*rdma_lite_deinit_mem_pool)(struct rdma_lite_context *lite_ctx, u32 mem_idx);

    struct rdma_lite_cq *(*rdma_lite_create_cq)(struct rdma_lite_context *lite_ctx,
        struct rdma_lite_cq_attr *lite_cq_attr);
    int (*rdma_lite_destroy_cq)(struct rdma_lite_cq *lite_cq);

    struct rdma_lite_qp *(*rdma_lite_create_qp)(struct rdma_lite_context *lite_ctx,
        struct rdma_lite_qp_attr *lite_qp_attr);
    int (*rdma_lite_destroy_qp)(struct rdma_lite_qp *lite_qp);

    int (*rdma_lite_poll_cq)(struct rdma_lite_cq *lite_cq, int num_entries, struct rdma_lite_wc *lite_wc);
    int (*rdma_lite_poll_cq_v2)(struct rdma_lite_cq *lite_cq, int num_entries, struct rdma_lite_wc_v2 *lite_wc);

    int (*rdma_lite_post_send)(struct rdma_lite_qp *lite_qp, struct rdma_lite_send_wr *wr,
        struct rdma_lite_send_wr **bad_wr, struct rdma_lite_post_send_attr *attr,
        struct rdma_lite_post_send_resp *resp);

    int (*rdma_lite_post_recv)(struct rdma_lite_qp *lite_qp, struct rdma_lite_recv_wr *wr,
        struct rdma_lite_recv_wr **bad_wr);

    int (*rdma_lite_set_qp_sl)(struct rdma_lite_qp *lite_qp, int sl);
    int (*rdma_lite_clean_qp)(struct rdma_lite_qp *lite_qp);
};

extern struct rdma_lite_ops g_hns_roce_lite_ops;

struct rdma_lite_ops *get_hns_roce_lite_ops(void);

#endif /* RDMA_LITE_H */