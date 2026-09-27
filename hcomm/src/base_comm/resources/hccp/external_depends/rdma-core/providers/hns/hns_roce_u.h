/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _HNS_ROCE_U_H
#define _HNS_ROCE_U_H

#include <infiniband/driver.h>
#include <util/udma_barrier.h>
#include <ccan/container_of.h>
#include "verbs_exp.h"
#include "peer_ops.h"
#ifndef HNS_ROCE_LLT
#include "dlog_pub.h"
#endif
#include "user_log.h"

#if defined(HNS_ROCE_LLT) || defined(DEFINE_HNS_LLT)
#define STATIC
#else
#define STATIC static
#endif

#define HNS_ROCE_HW_VER1 ('h' << 24 | 'i' << 16 | '0' << 8 | '6')

#define HNS_ROCE_HW_VER2 ('h' << 24 | 'i' << 16 | '0' << 8 | '8')

#define HNS_ROCE_HW_VER3 ('h' << 24 | 'i' << 16 | '0' << 8 | '9')

#define PFX "hns: "

#define RAW_FW_VERSION_32 32
#define RAW_FW_VERSION_16 16

#define HNS_ROCE_DWQE_PAGE_SIZE 65536
#define HNS_ROCE_MAX_INLINE_DATA_LEN 32
#define HNS_ROCE_V3_MAX_INLINE_DATA_LEN 1024
#define HNS_ROCE_MAX_RC_INL_INN_SZ 32
#define HNS_ROCE_MAX_CQ_NUM 0x10000
#define HNS_ROCE_MAX_SRQWQE_NUM 0x8000
#define HNS_ROCE_MAX_SRQSGE_NUM 0x100
#define HNS_ROCE_MIN_CQE_NUM 0x40
#define HNS_ROCE_MIN_WQE_NUM 0x20

#define HNS_ROCE_CQE_ENTRY_SIZE 0x20
#define HNS_ROCE_SQWQE_SHIFT 6
#define HNS_ROCE_SGE_IN_WQE 2
#define HNS_ROCE_SGE_SIZE 16
#define HNS_ROCE_SGE_SHIFT 4

#define HNS_ROCE_GID_SIZE 16

#define HNS_ROCE_CQ_DB_BUF_SIZE ((HNS_ROCE_MAX_CQ_NUM >> 11) << 12)
#define HNS_ROCE_TPTR_OFFSET 0x1000
#ifdef HNS_ROCE_DEVICE
#define HNS_ROCE_SHARE_BUF_SIZE 0x4000000
#else
#define HNS_ROCE_SHARE_BUF_SIZE 0x200000
#endif
#define HNS_ROCE_SHARED_MAX_SQ_NUM 4096
// share offset move to page 2, notify use page 1
#define HNS_ROCE_SHARE_OFFSET 0x3000
#define HNS_ROCE_NOTIFY_BUF_SIZE (1 << 13)
#define HNS_ROCE_NOTIFY_OFFSET 0x1000

#define HNS_ROCE_STATIC_RATE 3 /* Gbps */

#define HNS_ROCE_DFX_RESERVER_SIEZE 26
#define WQE_SHIFT_START 6
#define DB_PI_OFFSET 32  /* producer index offset */
#define DB_QPN_OFFSET 24 /* QPN offset */

#define roce_get_field(origin, mask, shift) (((le32toh(origin)) & (mask)) >> (shift))

#define roce_get_bit(origin, shift) (unsigned int)roce_get_field((origin), (1ul << (shift)), (shift))

#define roce_set_field(origin, mask, shift, val)                                                                       \
    do {                                                                                                               \
        (origin) &= ~htole32(mask);                                                                                    \
        (origin) |= htole32(((unsigned int)((val) << (shift))) & (mask));                                              \
    } while (0)

#define roce_set_bit(origin, shift, val) roce_set_field((origin), (unsigned long)(1UL << (shift)), (shift), (val))

enum {
    HNS_ROCE_QP_TABLE_BITS = 8,
    HNS_ROCE_QP_TABLE_SIZE = 1 << HNS_ROCE_QP_TABLE_BITS,
};

/* operation type list */
enum {
    /* rq&srq operation */
    HNS_ROCE_OPCODE_SEND_DATA_RECEIVE = 0x06,
    HNS_ROCE_OPCODE_RDMA_WITH_IMM_RECEIVE = 0x07,
};

/* qp mode */
#define HNS_ROCE_QP_MODE_NOR 0 // common qp
#define HNS_ROCE_QP_MODE_GDR 1 // GDR qp
#define HNS_ROCE_QP_MODE_OP 2  // op
#define HNS_ROCE_QP_MODE_ERR 3 // ERROR

enum { HNS_ROCE_NOT_SUPPORT_LITE_OP = 0, HNS_ROCE_SUPPORT_LITE_OP = 1 };

struct hns_roce_create_type {
    int type;
    int lite_op_supported;
    int mem_align; // 0,1:4KB, 2:2MB
    unsigned int mem_idx;
};

struct hns_roce_device {
    struct verbs_device ibv_dev;
    int page_size;
    struct hns_roce_u_hw *u_hw;
    int hw_version;
};

struct hns_roce_buf {
    // output attrs
    void *buf;
    unsigned int length;
    unsigned int offset;
    // input attrs
    int mem_align;
    unsigned int mem_idx;
};

struct hns_roce_mem_pool {
    struct hns_roce_buf roce_buf;
    unsigned int used_size;
};

#define BIT_CNT_PER_BYTE 8

/* the sw db length, on behalf of the qp/cq/srq length from left to right; */
static const unsigned int g_db_size[] = {4, 4};

/* the sw doorbell type; */
enum hns_roce_db_type { HNS_ROCE_QP_TYPE_DB, HNS_ROCE_CQ_TYPE_DB, HNS_ROCE_DB_TYPE_NUM };

struct hns_roce_db_page {
    struct hns_roce_db_page *prev, *next;
    struct hns_roce_buf buf;
    unsigned int num_db;
    unsigned int use_cnt;
    unsigned long *bitmap;
    unsigned int buf_type;
};

/* create qp/cq type; */
enum {
    HNS_ROCE_CREATE_NOR, // Normal
    HNS_ROCE_CREATE_EXP, // GDR
    HNS_ROCE_CREATE_OP   // OP
};

enum hns_roce_buf_type {
    HNS_ROCE_BUF_TYPE_NORMAL = 0,
    HNS_ROCE_BUF_TYPE_LITE_ALIGN_4KB = 1,
    HNS_ROCE_BUF_TYPE_LITE_ALIGN_2MB = 2
};

#define LITE_ALIGN_2MB (1 << 1)

struct hns_roce_context {
    struct verbs_context ibv_ctx;
    struct verbs_context_exp ibv_ctx_exp;
    void *uar;
    void *dwqe_page;
    pthread_spinlock_t uar_lock;

    void *cq_tptr_base;

    void *share_buffer_base;
    unsigned int port_num;
    unsigned int port_id;
    unsigned int dev_id;
    void *notify_va_base;
    void *notify_pa_base;
    unsigned long long notify_size;

    struct {
        struct hns_roce_qp **table;
        int refcnt;
    } qp_table[HNS_ROCE_QP_TABLE_SIZE];

    pthread_mutex_t qp_table_mutex;

    int num_qps;
    int qp_table_shift;
    int qp_table_mask;

    struct hns_roce_db_page *db_list[HNS_ROCE_DB_TYPE_NUM];
    pthread_mutex_t db_list_mutex;

    unsigned int max_qp_wr;
    unsigned int max_sge;
    int max_cqe;
};

struct hns_roce_pd {
    struct ibv_pd ibv_pd;
    unsigned int pdn;
};

struct hns_roce_cq {
    struct verbs_cq verbs_cq;
    struct hns_roce_buf buf;
    pthread_spinlock_t lock;
    unsigned int cqn;
    unsigned int cq_depth;
    unsigned int cons_index;
    unsigned int *set_ci_db;
    unsigned int *arm_db;
    int arm_sn;
    unsigned int flags;
    struct list_head list_sq;
    struct list_head list_rq;
    enum hns_roce_buf_type buf_type;
};

struct hns_roce_idx_que {
    struct hns_roce_buf buf;
    int buf_size;
    int entry_sz;
    unsigned long *bitmap;
    unsigned long use_cnt;
    int bitmap_len;
};

struct hns_roce_srq {
    struct verbs_srq verbs_srq;
    struct hns_roce_buf buf;
    pthread_spinlock_t lock;
    unsigned long *wrid;
    unsigned int srqn;
    int max;
    unsigned int max_gs;
    int wqe_shift;
    int head;
    int tail;
    unsigned int *db;
    unsigned short counter;
    struct hns_roce_idx_que idx_que;
};

struct hns_roce_wq {
    unsigned long *wrid;
    unsigned int wrid_len;
    pthread_spinlock_t lock;
    unsigned int wqe_cnt;
    int max_post;
    unsigned int head;
    unsigned int tail;
    unsigned int max_gs;
    int wqe_shift;
    unsigned int shift; /* wq size is 2^shift */
    int offset;
};

struct hns_roce_sge_ex {
    int offset;
    unsigned int sge_cnt;
    int sge_shift;
};

struct hns_roce_rinl_sge {
    void *addr;
    unsigned int len;
};

struct hns_roce_rinl_wqe {
    struct hns_roce_rinl_sge *sg_list;
    unsigned int sge_cnt;
};

struct hns_roce_rinl_buf {
    struct hns_roce_rinl_wqe *wqe_list;
    unsigned int wqe_cnt;
};

enum { HNS_QP_PEER_VA_ID_DBR = 0, HNS_QP_PEER_VA_ID_MAX = 1 };

enum {
    CREATE_FLAG_NO_DOORBELL = 1 << 2,
};

struct hns_roce_dfx_context {
    unsigned int sq_index;
    unsigned int sq_pi;
    unsigned int wq_temp_index;
    unsigned int sream_id;
    uint64_t db_value;
    unsigned int resv[HNS_ROCE_DFX_RESERVER_SIEZE];
};

struct hns_roce_qp {
    struct ibv_qp ibv_qp;
    struct hns_roce_buf buf;
    unsigned int max_inline_data;
    int buf_size;
    unsigned int sq_signal_bits;
    struct hns_roce_wq sq;
    struct hns_roce_wq rq;
    unsigned int *rdb;
    unsigned int *sdb;
    struct hns_roce_sge_ex sge;
    unsigned int next_sge;
    int port_num;
    int sl;

    struct hns_roce_rinl_buf rq_rinl_buf;
    unsigned int flags;
    struct list_node rcq_list;
    struct list_node scq_list;

    int peer_enabled;
    uint64_t peer_va_ids[HNS_QP_PEER_VA_ID_MAX];
    uint16_t create_flags;
    uint64_t peer_ctrl_db;

    int gdr_enabled;
    enum hns_roce_buf_type buf_type;
    struct ibv_exp_gdr_share_sq gdr_share_sq;
    struct ibv_exp_gdr_temp_wqe gdr_temp_wqe;
};

#define MAC_LEN 6
struct hns_roce_av {
    __le32 port_pd;
    uint8_t gid_index;
    uint8_t stat_rate;
    uint8_t hop_limit;
    __le16 sl;
    __le16 tclass;
    __le32 flowlabel;
    uint8_t dgid[HNS_ROCE_GID_SIZE];
    uint8_t mac[MAC_LEN];
    __le16 vlan;
};

struct hns_roce_ah {
    struct ibv_ah ibv_ah;
    struct hns_roce_av av;
};

struct hns_roce_u_hw {
    uint32_t hw_version;
    struct verbs_context_ops hw_ops;
    int (*exp_peer_commit_qp)(struct ibv_qp *qp, struct ibv_exp_peer_commit *peer);
    int (*exp_post_send)(struct ibv_qp *ibvqp, struct ibv_send_wr *wr, struct ibv_send_wr **bad_wr,
        struct wr_exp_rsp *exp_rsp);
};

static inline unsigned long hns_roce_align(unsigned long val, unsigned long align)
{
    return (val + align - 1) & ~(align - 1);
}

static inline struct hns_roce_device *to_hr_dev(struct ibv_device *ibv_dev)
{
    return container_of(ibv_dev, struct hns_roce_device, ibv_dev.device);
}

static inline struct hns_roce_context *to_hr_ctx(struct ibv_context *ibv_ctx)
{
    return container_of(ibv_ctx, struct hns_roce_context, ibv_ctx.context);
}

static inline struct hns_roce_pd *to_hr_pd(struct ibv_pd *ibv_pd)
{
    return container_of(ibv_pd, struct hns_roce_pd, ibv_pd);
}

static inline struct hns_roce_cq *to_hr_cq(struct ibv_cq *ibv_cq)
{
    return container_of(ibv_cq, struct hns_roce_cq, verbs_cq.cq);
}

static inline struct hns_roce_srq *to_hr_srq(struct ibv_srq *ibv_srq)
{
    struct verbs_srq *srq_temp = container_of(ibv_srq, struct verbs_srq, srq);
    return container_of(srq_temp, struct hns_roce_srq, verbs_srq);
}

static inline struct hns_roce_qp *to_hr_qp(struct ibv_qp *ibv_qp)
{
    return container_of(ibv_qp, struct hns_roce_qp, ibv_qp);
}

static inline struct hns_roce_ah *to_hr_ah(struct ibv_ah *ibv_ah)
{
    return container_of(ibv_ah, struct hns_roce_ah, ibv_ah);
}

int hns_roce_u_query_device(struct ibv_context *context, const struct ibv_query_device_ex_input *input,
    struct ibv_device_attr_ex *attr, size_t attr_size);
int hns_roce_u_query_port(struct ibv_context *context, uint8_t port, struct ibv_port_attr *attr);

struct ibv_pd *hns_roce_u_alloc_pd(struct ibv_context *context);
int hns_roce_u_free_pd(struct ibv_pd *pd);

struct ibv_mr *hns_roce_u_reg_mr(struct ibv_pd *pd, void *addr, size_t length, uint64_t hca_va, int access);
struct ibv_mr *hns_roce_u_exp_reg_mr(struct ibv_pd *pd, void *addr, size_t length, int access,
    struct roce_process_sign roce_sign);
int hns_roce_u_rereg_mr(struct verbs_mr *vmr, int flags, struct ibv_pd *pd, void *addr, size_t length, int access);
int hns_roce_u_dereg_mr(struct verbs_mr *vmr);

struct ibv_mw *hns_roce_u_alloc_mw(struct ibv_pd *pd, enum ibv_mw_type type);
int hns_roce_u_dealloc_mw(struct ibv_mw *mw);
int hns_roce_u_bind_mw(struct ibv_qp *qp, struct ibv_mw *mw, struct ibv_mw_bind *mw_bind);

struct ibv_cq *hns_roce_u_create_cq(struct ibv_context *context, int cqe, struct ibv_comp_channel *channel, int vector);

int hns_roce_u_modify_cq(struct ibv_cq *cq, struct ibv_modify_cq_attr *attr);
int hns_roce_u_destroy_cq(struct ibv_cq *cq);
void hns_roce_u_cq_event(struct ibv_cq *cq);

struct ibv_srq *hns_roce_u_create_srq(struct ibv_pd *pd, struct ibv_srq_init_attr *srq_init_attr);
int hns_roce_u_modify_srq(struct ibv_srq *srq, struct ibv_srq_attr *srq_attr, int srq_attr_mask);
int hns_roce_u_query_srq(struct ibv_srq *srq, struct ibv_srq_attr *srq_attr);
int hns_roce_u_destroy_srq(struct ibv_srq *srq);
struct ibv_qp *hns_roce_u_create_qp(struct ibv_pd *pd, struct ibv_qp_init_attr *attr);

struct ibv_qp *hns_roce_u_exp_create_qp(struct ibv_pd *pd, struct ibv_exp_qp_init_attr *attrx,
    struct rdma_lite_device_qp_attr *qp_resp);

struct ibv_cq *hns_roce_u_exp_create_cq(struct ibv_context *context, int cqe, struct ibv_comp_channel *channel,
    int vector, struct rdma_lite_device_cq_init_attr *attr, struct rdma_lite_device_cq_attr *cq_resp);

int hns_roce_u_exp_query_notify(struct ibv_context *ibv_context, unsigned long long *notify_va,
    unsigned long long *size);

int hns_roce_u_query_qp(struct ibv_qp *ibqp, struct ibv_qp_attr *attr, int attr_mask,
    struct ibv_qp_init_attr *init_attr);

struct ibv_ah *hns_roce_u_create_ah(struct ibv_pd *pd, struct ibv_ah_attr *attr);
int hns_roce_u_destroy_ah(struct ibv_ah *ah);

int hns_roce_alloc_buf(struct hns_roce_buf *buf, unsigned int size, int page_size);

int hns_roce_hal_alloc_buf(struct hns_roce_buf *buf, unsigned int size, unsigned int page_size, unsigned int dev_id);

void hns_roce_free_buf(struct hns_roce_buf *buf);

int roce_init_mem_pool(const struct roce_mem_cq_qp_attr *mem_attr, struct rdma_lite_device_mem_attr *mem_data,
    unsigned int dev_id);

int hns_roce_hal_alloc_mem(struct hns_roce_buf *buf, unsigned int size, unsigned int page_size);

int hns_roce_hal_free_mem(struct hns_roce_buf *buf);

int roce_deinit_mem_pool(unsigned int mem_idx);

void hns_roce_init_qp_indices(struct hns_roce_qp *qp);

int hns_roce_u_exp_query_device(struct ibv_context *ibv_context, struct dev_cap_info *cap);

extern struct hns_roce_u_hw g_hns_roce_u_hw_v2;

extern struct hns_roce_u_hw g_hns_roce_u_hw_v3;

#endif /* _HNS_ROCE_U_H */
