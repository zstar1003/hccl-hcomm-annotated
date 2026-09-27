/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _HNS_ROCE_U_HW_V2_QP_H
#define _HNS_ROCE_U_HW_V2_QP_H
#include "peer_ops.h"
#include "hns_roce_u.h"

#define INDEX_LEN 11
#define INFO_PAYLOAD_LEN 1000
struct hns_roce_qpc_stat {
    char index[INDEX_LEN];
    char info[INFO_PAYLOAD_LEN];
};

void hns_roce_u_v2_printf_dfx(struct hns_roce_qp *qp, unsigned int opcode);
int hns_roce_u_v2_modify_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr, int attr_mask);
int hns_roce_u_v2_destroy_qp(struct ibv_qp *ibqp);
int hns_roce_u_v2_exp_peer_commit_qp(struct ibv_qp *ibvqp, struct ibv_exp_peer_commit *commit_ctx);

#endif /* _HNS_ROCE_U_HW_V2_QP_H */
