/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HNS_ROCE_U_CMD_H
#define HNS_ROCE_U_CMD_H

#include <unistd.h>
#include "hns_roce_u_hw_v2_qp.h"

/**
 * API VERSION NUMBER combines major version, minor version and patch version, version range form 0x00 to 0xff.
 * example : 0x020103 means version 0x020103, major 0x02, minor 0x01, patch 0x03
 * when delete API, modify API name, should add major version.
 * when add new API, should add minor version.
 * when modify enum para, struct para add patch version. this means when new API compatible with old API
 */
#define ROCE_API_VER_MAJOR 0x0
#define ROCE_API_VER_MINOR 0x0
#define ROCE_API_VER_PATCH 0x0
#define ROCE_API_VERSION ((ROCE_API_VER_MAJOR << 16U) | (ROCE_API_VER_MINOR << 8U) | (ROCE_API_VER_PATCH))

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

struct hns_roce_mr_remap_info {
    void *va;                /**< starting address need to remap of mr */
    unsigned long long size; /**< size need to remap of mr */
};

#define ROCE_IOCTL_MAGIC 'R'
#define ROCE_CMD_SET_TSQP_DEPTH _IO(ROCE_IOCTL_MAGIC, 1)
#define ROCE_CMD_GET_TSQP_DEPTH _IO(ROCE_IOCTL_MAGIC, 2)
#define ROCE_CMD_GET_ROCE_DEV_INFO _IO(ROCE_IOCTL_MAGIC, 3)
#define ROCE_CMD_GET_ROCE_QPC_STAT _IO(ROCE_IOCTL_MAGIC, 4)

int roce_set_tsqp_depth(const char *dev_name, unsigned int rdev_index, unsigned int temp_depth, unsigned int *qp_num,
    unsigned int *sq_depth);
int roce_get_tsqp_depth(const char *dev_name, unsigned int rdev_index, unsigned int *temp_depth, unsigned int *qp_num,
    unsigned int *sq_depth);
int roce_get_roce_dev_data(const char *dev_name, struct roce_dev_data *dev_data);
int hns_roce_u_get_roce_qpc_stat(const struct ibv_context *ibv_ctx, struct hns_roce_qpc_stat *qpc_stat);
int roce_query_qpc(struct ibv_qp *qp, struct hns_roce_qpc_attr_val *attr_val, unsigned int attr_mask);
int roce_mmap_ai_db_reg(struct ibv_context *ibv_ctx, unsigned int tgid);
int roce_unmmap_ai_db_reg(struct ibv_context *ibv_ctx);
int roce_get_cq_data_plane_info(struct ibv_cq *cq, struct hns_roce_cq_data_plane_info *info);
int roce_get_qp_data_plane_info(struct ibv_qp *qp, struct hns_roce_qp_data_plane_info *info);
int roce_remap_mr(struct ibv_mr *ibvmr, struct hns_roce_mr_remap_info info[], unsigned int num);
unsigned int roce_get_api_version(void);
#endif // HNS_ROCE_U_CMD_H
