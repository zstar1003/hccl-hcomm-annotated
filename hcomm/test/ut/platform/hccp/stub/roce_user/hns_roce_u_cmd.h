/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef STUB_HNS_ROCE_USER_CDEV_H
#define STUB_HNS_ROCE_USER_CDEV_H
#include <stdio.h>
#include <unistd.h>

#include "rs_inner.h"

struct roce_set_tsqp_depth_data {
    unsigned int rdev_index;
    unsigned int temp_depth;
    unsigned int sq_depth;
    unsigned int qp_num;
};

struct roce_get_tsqp_depth_data {
    unsigned int rdev_index;
    unsigned int temp_depth;
    unsigned int sq_depth;
    unsigned int qp_num;
};

struct roce_dev_data {
    unsigned int rdev_index;
    unsigned int reserved;
};

enum hns_roce_ai_qpc_attr_mask {
    HNS_ROCE_AI_QPC_UDPSPN = 1 << 0,
};

struct hns_roce_qpc_attr_val {
    unsigned int udp_sport;
};

struct hns_roce_qpc_attr {
    unsigned int qpn;
    struct hns_roce_qpc_attr_val attr_val;
    unsigned int attr_mask;
};

struct hns_roce_wq_data_plane_info {
    unsigned int wqn;
    unsigned long long buf_addr;
    unsigned int wqebb_size;
    unsigned int depth;
    unsigned long long head_addr;
    unsigned long long tail_addr;
    unsigned long long swdb_addr;
    unsigned long long db_reg;
};

struct hns_roce_cq_data_plane_info {
    unsigned int cqn;
    unsigned long long buf_addr;
    unsigned int cqe_size;
    unsigned int depth;
    unsigned long long head_addr;
    unsigned long long tail_addr;
    unsigned long long swdb_addr;
    unsigned long long db_reg;
};

struct hns_roce_qp_data_plane_info {
    struct hns_roce_wq_data_plane_info sq;
    struct hns_roce_wq_data_plane_info rq;
};

struct ibv_exp_ah_attr {
    struct ibv_ah_attr attr;
    uint32_t udp_sport;
};

struct hns_roce_mr_remap_info {
    void* va;                /**< starting va need to remap of mr */
    unsigned long long size; /**< size need to remap of mr */
};

#define ROCE_IOCTL_MAGIC 'R'
#define ROCE_CMD_SET_TSQP_DEPTH _IO(ROCE_IOCTL_MAGIC, 1)
#define ROCE_CMD_GET_TSQP_DEPTH _IO(ROCE_IOCTL_MAGIC, 2)
#define ROCE_CMD_GET_ROCE_DEV_INFO _IO(ROCE_IOCTL_MAGIC, 3)

int roce_set_tsqp_depth(
    const char* dev_name, unsigned int rdev_index, unsigned int temp_depth, unsigned int* qp_num,
    unsigned int* sq_depth);
int roce_get_tsqp_depth(
    const char* dev_name, unsigned int rdev_index, unsigned int* temp_depth, unsigned int* qp_num,
    unsigned int* sq_depth);
int roce_get_roce_dev_data(const char* dev_name, struct roce_dev_data* dev_data);
int roce_init_mem_pool(
    const struct roce_mem_cq_qp_attr* mem_attr, struct rdma_lite_device_mem_attr* mem_data, unsigned int dev_id);
int roce_deinit_mem_pool(unsigned int mem_idx);
int roce_query_qpc(struct ibv_qp* qp, struct hns_roce_qpc_attr_val* attr_val, unsigned int attr_mask);
struct ibv_ah* ibv_exp_create_ah(struct ibv_pd* pd, struct ibv_exp_ah_attr* attrx);
int roce_mmap_ai_db_reg(struct ibv_context* ibv_ctx, unsigned int tgid);
int roce_unmmap_ai_db_reg(struct ibv_context* ibv_ctx);
int roce_get_cq_data_plane_info(struct ibv_cq* cq, struct hns_roce_cq_data_plane_info* info);
int roce_get_qp_data_plane_info(struct ibv_qp* qp, struct hns_roce_qp_data_plane_info* info);
int roce_remap_mr(struct ibv_mr* mr, struct hns_roce_mr_remap_info info[], unsigned int num);
unsigned int roce_get_api_version(void);

int roce_set_qp_lb_value(struct ibv_qp* qp, int lb_value);
int roce_get_qp_lb_value(struct ibv_qp* qp, int* lb_value);
int roce_get_qp_num(struct ibv_context* context, int* qp_num);
#endif
