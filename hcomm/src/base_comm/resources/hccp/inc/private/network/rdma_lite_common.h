/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RDMA_LITE_COMMON_H
#define RDMA_LITE_COMMON_H

struct rdma_lite_device_cq_init_attr {
    int cq_type;
    int part_id;
    int lite_op_supported;
    int mem_align; // 0,1:4KB, 2:2MB
    unsigned int mem_idx;
    unsigned int ai_op_support;
    unsigned int grp_id;
    unsigned int cq_cstm_flag;
    unsigned int use_resv_mem;
    unsigned int resv_mem_pool_id;
    unsigned int resv[16U];
};

struct rdma_lite_device_buf {
    unsigned long long va;
    unsigned int len;
};

struct rdma_lite_device_cq_attr {
    unsigned int depth;                   // 创建的CQ深度
    unsigned int flags;                   // 是否支持Record DB，1:支持，0:不支持
    unsigned int cqe_size;                // CQE大小
    unsigned int cqn;                     // CQ编号
    struct rdma_lite_device_buf cq_buf;   // CQ buf的Device侧VA和 len
    struct rdma_lite_device_buf swdb_buf; // CQ软件DB的Device侧VA 和 len
};

struct rdma_lite_device_mem_attr {
    unsigned long long va;
    unsigned int mem_size;
    unsigned int mem_idx;
};

struct rdma_lite_device_wqe {
    int max_post;
    unsigned int max_gs;
    unsigned int wqe_cnt;
    unsigned int wqe_shift;
    int offset;
    unsigned int wrid_len;
    unsigned int shift; /* wq size is 2^shift */
    struct rdma_lite_device_buf db_buf;
};

struct rdma_lite_device_sge {
    int offset;
    unsigned int sge_cnt;
    int sge_shift;
};

struct rdma_lite_device_qp_attr {
    unsigned int max_inline_data;
    unsigned int next_sge;
    int qp_info;
    unsigned int qpn;
    struct rdma_lite_device_buf qp_buf;
    struct rdma_lite_device_wqe sq;
    struct rdma_lite_device_wqe rq;
    struct rdma_lite_device_sge sge;
};

struct roce_mem_cq_qp_attr {
    unsigned int send_cq_depth;
    unsigned int send_qp_depth;
    unsigned int send_sge_num;
    unsigned int recv_cq_depth;
    unsigned int recv_qp_depth;
    unsigned int recv_sge_num;
    int mem_align; // 0,1:4KB, 2:2MB
    unsigned int use_resv_mem;
    unsigned int resv_mem_pool_id;
    unsigned int resv0;
    void *ctx; // refer to struct ibv_context
    unsigned int resv[16U];
};

struct dev_cap_info {
    unsigned int num_qps;  // 设备可以使用的QP数量
    unsigned int port_num; // 设备有几个物理端口
    int qp_table_shift;
    int qp_table_mask;
    unsigned int max_qp_wr;
    unsigned int max_sge;
    unsigned int page_size;
};

#endif /* RDMA_LITE_COMMON_H */