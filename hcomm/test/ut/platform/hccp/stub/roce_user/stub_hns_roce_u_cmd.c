/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dl_ibverbs_function.h"
#include "hns_roce_u_cmd.h"

int roce_set_tsqp_depth(
    const char* dev_name, unsigned int rdev_index, unsigned int temp_depth, unsigned int* qp_num,
    unsigned int* sq_depth)
{
    return 0;
}

int roce_get_tsqp_depth(
    const char* dev_name, unsigned int rdev_index, unsigned int* temp_depth, unsigned int* qp_num,
    unsigned int* sq_depth)
{
    *sq_depth = 128;
    *temp_depth = 12;
    *qp_num = 1024;

    return 0;
}

int roce_get_roce_dev_data(const char* dev_name, struct roce_dev_data* dev_data)
{
    dev_data->rdev_index = 0;

    return 0;
}

int roce_init_mem_pool(
    const struct roce_mem_cq_qp_attr* mem_attr, struct rdma_lite_device_mem_attr* mem_data, unsigned int dev_id)
{
    return 0;
}

int roce_deinit_mem_pool(unsigned int mem_idx) { return 0; }

int roce_query_qpc(struct ibv_qp* qp, struct hns_roce_qpc_attr_val* attr_val, unsigned int attr_mask) { return 0; }

struct ibv_ah* ibv_exp_create_ah(struct ibv_pd* pd, struct ibv_exp_ah_attr* attrx) { return NULL; }

int roce_mmap_ai_db_reg(struct ibv_context* ibv_ctx, unsigned int tgid) { return 0; }

int roce_unmmap_ai_db_reg(struct ibv_context* ibv_ctx) { return 0; }

int roce_get_cq_data_plane_info(struct ibv_cq* cq, struct hns_roce_cq_data_plane_info* info) { return 0; }

int roce_get_qp_data_plane_info(struct ibv_qp* qp, struct hns_roce_qp_data_plane_info* info) { return 0; }

int roce_remap_mr(struct ibv_mr* mr, struct hns_roce_mr_remap_info info[], unsigned int num) { return 0; }

unsigned int roce_get_api_version(void) { return 0; }

int roce_set_qp_lb_value(struct ibv_qp* qp, int lb_value) { return 0; }

int roce_get_qp_lb_value(struct ibv_qp* qp, int* lb_value) { return 0; }

int roce_get_qp_num(struct ibv_context* context, int* qp_num)
{
    (void)context;
    return 0;
}
