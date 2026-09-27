/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HNS_ROCE_LITE_H
#define HNS_ROCE_LITE_H

#include "verbs_exp.h"
#include "rdma_lite.h"
#include "ascend_hal.h"
#include "hns_roce_u_hw_v2.h"

#define BIT_CNT_PER_BYTE       8
#define BIT_CNT_PER_LONG       (BIT_CNT_PER_BYTE * sizeof(uint64_t))

#define GENMASK(h, l)          (((~0UL) - (1UL << (l)) + 1) & (~0UL >> (BIT_CNT_PER_LONG - 1 - (h))))

#define RECORD_DB_CI_MASK GENMASK(23, 0)

#define HNS_ROCE_LITE_CQ_FLAG_RECORD_DB  (1 << 0)
#define HNS_ROCE_LITE_CQE_SUCCESS        0
#define HNS_ROCE_LITE_CQE_IS_SQ          0

#define HNS_ROCE_LITE_INVALID_SGE_KEY    0x100

enum {
    HNS_ROCE_LITE_CQE_QPN_MASK    = 0x3ffff,
    HNS_ROCE_LITE_CQE_STATUS_MASK = 0xff,
    HNS_ROCE_LITE_CQE_OPCODE_MASK = 0x1f
};

#define CQE_BYTE_4_OPCODE_S 0
#define CQE_BYTE_4_OPCODE_M   (((1UL << 5) - 1) << CQE_BYTE_4_OPCODE_S)

#define CQE_BYTE_4_RQ_INLINE_S 5

#define CQE_BYTE_4_S_R_S 6
#define CQE_BYTE_4_OWNER_S 7

#define CQE_BYTE_4_STATUS_S 8
#define CQE_BYTE_4_STATUS_M   (((1UL << 8) - 1) << CQE_BYTE_4_STATUS_S)

#define CQE_BYTE_4_WQE_IDX_S 16
#define CQE_BYTE_4_WQE_IDX_M   (((1UL << 16) - 1) << CQE_BYTE_4_WQE_IDX_S)

#define CQE_BYTE_12_XRC_SRQN_S 0
#define CQE_BYTE_12_XRC_SRQN_M   (((1UL << 24) - 1) << CQE_BYTE_12_XRC_SRQN_S)

#define CQE_BYTE_16_LCL_QPN_S 0
#define CQE_BYTE_16_LCL_QPN_M   (((1UL << 24) - 1) << CQE_BYTE_16_LCL_QPN_S)

#define CQE_BYTE_28_SMAC_S 0
#define CQE_BYTE_28_SMAC_M   (((1UL << 16) - 1) << CQE_BYTE_28_SMAC_S)

#define CQE_BYTE_28_PORT_TYPE_S 16
#define CQE_BYTE_28_PORT_TYPE_M   (((1UL << 2) - 1) << CQE_BYTE_28_PORT_TYPE_S)

#define CQE_BYTE_32_RMT_QPN_S 0
#define CQE_BYTE_32_RMT_QPN_M   (((1UL << 24) - 1) << CQE_BYTE_32_RMT_QPN_S)

#define CQE_BYTE_32_SL_S 24
#define CQE_BYTE_32_SL_M   (((1UL << 3) - 1) << CQE_BYTE_32_SL_S)

#define CQE_BYTE_32_PORTN_S 27
#define CQE_BYTE_32_PORTN_M   (((1UL << 3) - 1) << CQE_BYTE_32_PORTN_S)

#define CQE_BYTE_32_GLH_S 30

#define CQE_BYTE_32_LPK_S 31

#define BYTE_LEN    8

#define DB_SL_OFFSET            48 /* service level offset */
#define DB_PI_OFFSET            32 /* producer index offset */
#define DB_QPN_OFFSET           24 /* QPN offset */

#define INLINE_REDUCE_OP_MASK                  0x7
#define INLINE_REDUCE_TYPE_MASK                0xf

#define RC_SQ_WQE_BYTE_20_REDUCE_TYPE_S        24

#define RC_SQ_WQE_BYTE_20_REDUCE_TYPE_M \
        (((1UL << 4) - 1) << RC_SQ_WQE_BYTE_20_REDUCE_TYPE_S)

#define RC_SQ_WQE_BYTE_20_REDUCE_OP_S 28
#define RC_SQ_WQE_BYTE_20_REDUCE_OP_M \
        (((1UL << 3) - 1) << RC_SQ_WQE_BYTE_20_REDUCE_OP_S)

#define RC_SQ_WQE_BYTE_12_NOTIFY_INFO_S 0
#define RC_SQ_WQE_BYTE_12_NOTIFY_INFO_M \
    (((1UL << 25) - 1) << RC_SQ_WQE_BYTE_12_NOTIFY_INFO_S)

#define RC_SQ_WQE_BYTE_4_OWNER_S 7
#define RC_SQ_WQE_BYTE_4_FENCE_S 9
#define RC_SQ_WQE_BYTE_4_SE_S 11
#define RC_SQ_WQE_BYTE_4_INLINE_S 12

#define RC_SQ_WQE_BYTE_16_SGE_NUM_S 24
#define RC_SQ_WQE_BYTE_16_SGE_NUM_M \
    (((1UL << 8) - 1) << RC_SQ_WQE_BYTE_16_SGE_NUM_S)

#define RC_SQ_WQE_BYTE_20_MSG_START_SGE_IDX_S 0
#define RC_SQ_WQE_BYTE_20_MSG_START_SGE_IDX_M \
    (((1UL << 24) - 1) << RC_SQ_WQE_BYTE_20_MSG_START_SGE_IDX_S)
#define RC_SQ_WQE_BYTE_20_INL_TYPE_S 31

#define RC_SQ_WQE_BYTE_4_OPCODE_S 0
#define RC_SQ_WQE_BYTE_4_OPCODE_M \
    (((1UL << 5) - 1) << RC_SQ_WQE_BYTE_4_OPCODE_S)

#define RC_SQ_WQE_BYTE_4_CQE_S 8
#define RC_MAX_SGE_NUME 2

// BITMAP_WORD_SIZE defined maximum supported size is 10K
#define BITMAP_WORD_LEN  32
#define BITMAP_WORD_NUM  320
#define BITMAP_WORD_SIZE (BITMAP_WORD_LEN * BITMAP_WORD_NUM)
#define PAGE_ALIGN_4KB   (4 * 1024)

enum hns_roce_qp_cap_flags {
    HNS_ROCE_QP_CAP_RQ_RECORD_DB = (1 << 0),
    HNS_ROCE_QP_CAP_SQ_RECORD_DB = (1 << 1),
    HNS_ROCE_QP_CAP_OWNER_DB     = (1 << 2),
    HNS_ROCE_QP_CAP_DIRECT_WQE   = (1 << 5)
};

enum {
    /* DFX info */
    HNS_ROCE_LITE_USER_CANCLE_QP = 0x0,
    HNS_ROCE_LITE_USER_POLL_CQE  = 0x1,
    HNS_ROCE_LITE_USER_RESERVER  = 0xff
};

enum {
    HNS_ROCE_LITE_QP_TABLE_BITS = 8,
    HNS_ROCE_LITE_QP_TABLE_SIZE = (1 << HNS_ROCE_LITE_QP_TABLE_BITS)
};

enum {
    HNS_ROCE_SQ_OP_NOP = 0x13,
    HNS_ROCE_SQ_OP_REDUCE_WRITE = 0x14,
    HNS_ROCE_SQ_OP_REDUCE_WRITE_NOTIFY = 0x15,
    HNS_ROCE_SQ_OP_WRITE_NOTIFY = 0x16,
    HNS_ROCE_SQ_OP_ATOMIC_WRITE = 0x17,
};

enum {
    HNS_ROCE_WC_MASK_COMP_SWAP = 8,
    HNS_ROCE_WC_MASK_FETCH_ADD,
    HNS_ROCE_WC_ATOMIC_WRITE = 0xf0,
    HNS_ROCE_WC_WRITE_NOTIFY = 0xf2,
    HNS_ROCE_WC_NOP = 0xf3,
    HNS_ROCE_WC_PERSIST_WRITE_WITH_IMM,
    HNS_ROCE_WC_REDUCE_WRITE,
    HNS_ROCE_WC_REDUCE_WRITE_NOTIFY,
};

struct hns_roce_lite_sge_ex {
    int offset;
    u32 sge_cnt;
    int sge_shift;
};

struct hns_roce_lite_sge_info {
    u32        valid_num; /* sge length is not 0 */
    u32        start_idx; /* start position of extend sge */
    u32        total_len; /* total length of msg length */
};

struct hns_roce_lite_wq {
    unsigned long               *wrid;  /* Work request ID */
    unsigned long               wrid_len;
    pthread_spinlock_t          lock;
    u32                         wqe_cnt;   /* WQE num */
    unsigned int                max_post;
    u32                         head;
    u32                         tail;
    u32                         max_gs;
    u32                         wqe_shift; /* WQE size */
    u32                         shift;     /* wq size is 2^shift */
    int                         offset;
};

struct hns_roce_lite_context {
    struct rdma_lite_context    lite_ctx;
    struct {
        struct hns_roce_lite_qp     **table; /* 参见struct hns_roce_lite_qp的定义 */
        int                         refcnt;
    } qp_table[HNS_ROCE_LITE_QP_TABLE_SIZE];
    pthread_mutex_t             qp_table_mutex;
    int                         qp_table_shift;
    int                         qp_table_mask;
    u32                         num_qps;

    u32                         max_qp_wr; /* QP最大容纳的WR个数 */
    u32                         max_sge;   /* QP最大的SGE个数(min(sq_sge_num, rq_sge_num)) */
    u32                         port_num;
    u32                         dev_id;
    u32                         page_size;
    struct list_head            db_list;
    u32                         mem_bitmap_list[BITMAP_WORD_NUM];
    struct rdma_lite_mem_pool   *mem_pool_list[BITMAP_WORD_SIZE];
    pthread_mutex_t             mutex;
};

struct hns_roce_lite_cq {
    struct rdma_lite_cq         lite_cq;
    u32                         depth;      /* 创建的CQ深度 */
    u32                         flags;      /* 是否支持Record DB，1:支持，0:不支持 */
    u32                         cqe_size;   /* CQE大小 */
    u32                         cqn;        /* CQ编号 */
    pthread_spinlock_t          lock;
    u32                         cons_index;
    struct rdma_lite_host_buf   swdb_buf;
    struct rdma_lite_host_buf   cq_buf;
    struct list_head            list_sq;
    struct list_head            list_rq;
};

struct hns_roce_lite_qp {
    struct rdma_lite_qp             lite_qp;
    struct rdma_lite_host_buf       buf;
    struct rdma_lite_host_buf       sdb_buf;
    struct rdma_lite_host_buf       rdb_buf;
    u32                             max_inline_data;
    int                             buf_size;
    u32                             sq_signal_bits;
    struct hns_roce_lite_wq         sq;
    struct hns_roce_lite_wq         rq;
    struct hns_roce_lite_sge_ex     sge;
    u32                             next_sge;
    int                             port_num;
    int                             sl;
    int                             gdr_enabled;
    u32                             flags;

    struct list_node                rcq_list;
    struct list_node                scq_list;
};

struct db_dva_node {
    u64                             db_align_dva;
    void                            *hva;
    struct list_node                entry;
    u32                             ref_cnt;
};

enum {
    CQ_OK            =  0,
    CQ_EMPTY         = -1,
    CQ_POLL_ERR      = -2
};

struct hns_roce_lite_cqe {
    __le32  byte_4;
    union {
        __le32  rkey;
        __le32  immtdata;
    };
    __le32  byte_12;
    __le32  byte_16;
    __le32  byte_cnt;
    __le32  smac;
    __le32  byte_28;
    __le32  byte_32;
};

struct hns_roce_lite_rc_sq_wqe {
    __le32  byte_4;
    __le32  msg_len;
    union {
        __le32  inv_key;
        __le32  immtdata;
        __le32  new_rkey;
    };
    __le32  byte_16;
    __le32  byte_20;
    __le32  rkey;
    __le64  va;
};

struct hns_roce_lite_wqe_data_seg {
    __le32      len;
    __le32      lkey;
    __le64      addr;
};

struct cqe_wc_status {
    u32 cqe_status;
    u32 wc_status;
};

enum hns_roce_lite_qp_ai_mode {
    HNS_ROCE_QP_AI_MODE_NOR         = 0,
    HNS_ROCE_QP_AI_MODE_GDR         = 1,
    HNS_ROCE_QP_AI_MODE_OP          = 2,
    HNS_ROCE_QP_AI_MODE_GDR_ASYN    = 3,
    HNS_ROCE_QP_AI_MODE_OP_EXT      = 4
};

static inline struct hns_roce_lite_cq *to_hr_lite_cq(struct rdma_lite_cq *lite_cq)
{
    return container_of(lite_cq, struct hns_roce_lite_cq, lite_cq);
}

static inline struct hns_roce_lite_qp *to_hr_lite_qp(struct rdma_lite_qp *lite_qp)
{
    return container_of(lite_qp, struct hns_roce_lite_qp, lite_qp);
}

static inline struct hns_roce_lite_context *to_hr_lite_ctx(struct rdma_lite_context *lite_ctx)
{
    return container_of(lite_ctx, struct hns_roce_lite_context, lite_ctx);
}

#define HR_RDMA_LITE_OPC_MAP(rdma_lite_key, hr_key) \
    [RDMA_LITE_WR_ ## rdma_lite_key] = HNS_ROCE_WQE_OP_ ## hr_key

static const u32 hns_roce_lite_op_code[] = {
    HR_RDMA_LITE_OPC_MAP(RDMA_WRITE, RDMA_WRITE),
    HR_RDMA_LITE_OPC_MAP(RDMA_WRITE_WITH_IMM, RDMA_WRITE_WITH_IMM),
    HR_RDMA_LITE_OPC_MAP(SEND, SEND),
    HR_RDMA_LITE_OPC_MAP(SEND_WITH_IMM, SEND_WITH_IMM),
    HR_RDMA_LITE_OPC_MAP(RDMA_READ, RDMA_READ),
    HR_RDMA_LITE_OPC_MAP(ATOMIC_CMP_AND_SWP, ATOMIC_COM_AND_SWAP),
    HR_RDMA_LITE_OPC_MAP(NOP, NOP),
    HR_RDMA_LITE_OPC_MAP(ATOMIC_WRITE, ATOMIC_WRITE),
    HR_RDMA_LITE_OPC_MAP(REDUCE_WRITE, REDUCE_WRITE),
    HR_RDMA_LITE_OPC_MAP(REDUCE_WRITE_NOTIFY, REDUCE_WRITE_NOTIFY),
    HR_RDMA_LITE_OPC_MAP(WRITE_WITH_NOTIFY, WRITE_WITH_NOTIFY),
};

#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))

static inline u32 to_hr_lite_opcode(enum rdma_lite_wr_opcode lite_opcode)
{
    if (lite_opcode >= ARRAY_SIZE(hns_roce_lite_op_code)) {
        return HNS_ROCE_WQE_OP_MASK;
    }

    return hns_roce_lite_op_code[lite_opcode];
}

struct rdma_lite_context *hns_roce_lite_alloc_context(u8 phy_id, struct dev_cap_info *cap);
void hns_roce_lite_free_context(struct rdma_lite_context *lite_ctx);

int hns_roce_lite_init_mem_pool(struct rdma_lite_context *lite_ctx, struct rdma_lite_mem_attr *lite_mem_attr);
int hns_roce_lite_deinit_mem_pool(struct rdma_lite_context *lite_ctx, u32 mem_idx);

struct rdma_lite_cq *hns_roce_lite_create_cq(struct rdma_lite_context *lite_ctx,
    struct rdma_lite_cq_attr *lite_cq_attr);
int hns_roce_lite_destroy_cq(struct rdma_lite_cq *lite_cq);

struct rdma_lite_qp *hns_roce_lite_create_qp(struct rdma_lite_context *lite_ctx,
    struct rdma_lite_qp_attr *lite_qp_attr);
int hns_roce_lite_destroy_qp(struct rdma_lite_qp *lite_qp);

int hns_roce_lite_poll_cq(struct rdma_lite_cq *lite_cq, int num_entries, struct rdma_lite_wc *lite_wc);
int hns_roce_lite_poll_cq_v2(struct rdma_lite_cq *lite_cq, int num_entries, struct rdma_lite_wc_v2 *lite_wc);

int hns_roce_lite_post_send(struct rdma_lite_qp *lite_qp, struct rdma_lite_send_wr *lite_wr,
    struct rdma_lite_send_wr **bad_wr, struct rdma_lite_post_send_attr *attr, struct rdma_lite_post_send_resp *resp);

int hns_roce_lite_post_recv(struct rdma_lite_qp *lite_qp, struct rdma_lite_recv_wr *lite_wr,
    struct rdma_lite_recv_wr **bad_wr);

int hns_roce_lite_set_qp_sl(struct rdma_lite_qp *lite_qp, int sl);

int hns_roce_lite_clean_qp(struct rdma_lite_qp *lite_qp);
#endif /* HNS_ROCE_LITE_H */
