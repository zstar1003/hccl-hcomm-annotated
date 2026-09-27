/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _ST_VERBS_EXP_H
#define _ST_VERBS_EXP_H

#include <infiniband/driver.h>

#define PROCESS_PSIZE_LENGTH 49
#define PROCESS_PRESV_LENGTH 4

struct roce_process_sign {
    int tgid;
    unsigned int devid; /* chipId */
    unsigned int vfid;
    char sign[PROCESS_PSIZE_LENGTH];
    char resv[PROCESS_PRESV_LENGTH];
};

struct ibv_exp_qp_init_attr {
    struct ibv_qp_init_attr attr;
    int gdr_enable;
    int lite_op_support;
    int mem_align;
    unsigned int mem_idx;
    unsigned int udp_sport;
    unsigned int ai_op_support;
    unsigned int grp_id;
    unsigned int qp_cstm_flag;
    unsigned int use_resv_mem;
    unsigned int resv_mem_pool_id;
    unsigned int resv[16U];
};

struct wr_exp_rsp {
    unsigned int wqe_index;
    unsigned long db_info;
};

struct ibv_qp* ibv_exp_create_qp(
    struct ibv_pd* pd, struct ibv_exp_qp_init_attr* qp_init_attr, struct rdma_lite_device_qp_attr* qp_resp);
int ibv_exp_post_send(
    struct ibv_qp* qp, struct ibv_send_wr* wr, struct ibv_send_wr** bad_wr, struct wr_exp_rsp* exp_rsp);

int ibv_exp_query_notify(struct ibv_context* context, unsigned long long* notify_va, unsigned long long* size);

struct ibv_mr*
ibv_exp_reg_mr(struct ibv_pd* pd, void* addr, size_t length, int access, struct roce_process_sign roce_sign);

struct ibv_post_send_ext_resp {
    unsigned int wqe_index;
    unsigned long db_info;
};

struct ibv_post_send_ext_attr {
    uint8_t reduce_op;
    uint8_t reduce_type;
};

int ibv_ext_post_send(
    struct ibv_qp* qp, struct ibv_send_wr* wr, struct ibv_send_wr** bad_wr, struct ibv_post_send_ext_attr* ext_attr,
    struct ibv_post_send_ext_resp* ext_resp);

struct ibv_cq* ibv_create_ext_cq(
    struct ibv_context* context, int cqe, void* cq_context, struct ibv_comp_channel* channel, int comp_vector,
    int partid);

struct ibv_cq* ibv_exp_create_cq(
    struct ibv_context* context, int cqe, void* cq_context, struct ibv_comp_channel* channel, int comp_vector,
    struct rdma_lite_device_cq_init_attr* attr, struct rdma_lite_device_cq_attr* cq_resp);

int ibv_exp_query_device(struct ibv_context* context, struct dev_cap_info* cap);
int ibv_exp_set_dev_id(struct ibv_context* context, unsigned int dev_id);
#endif
