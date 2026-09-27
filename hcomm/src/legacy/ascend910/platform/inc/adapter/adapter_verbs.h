/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_ADAPTER_VERBS_H
#define HCCL_ADAPTER_VERBS_H

#include <infiniband/verbs.h>
#include "hccl/base.h"
#include "dlhns_function.h"

namespace hccl {
HcclResult hrtIbvPostSrqRecv(struct ibv_srq* srq, struct ibv_recv_wr* wr, struct ibv_recv_wr** badWr);
HcclResult hrtIbvPostRecv(struct ibv_qp* qp, struct ibv_recv_wr* wr, struct ibv_recv_wr** badWr);
HcclResult hrtIbvPostSend(struct ibv_qp* qp, struct ibv_send_wr* wr, struct ibv_send_wr** badWr);
HcclResult hrtIbvPollCq(struct ibv_cq* cq, int maxNum, struct ibv_wc* wc, s32& num);
HcclResult hrtIbvReqNotifyCq(struct ibv_cq* cq, int solicitedOnly);
HcclResult hrtIbvGetCqEvent(struct ibv_comp_channel* channel, struct ibv_cq** cq, void** cq_context);
void hrtIbvAckCqEvent(struct ibv_cq* qp, unsigned int nevents);
HcclResult hrtIbvQueryQp(struct ibv_qp* qp);
HcclResult
HrtHnsIbvExpPostSend(struct ibv_qp* qp, struct ibv_send_wr* wr, struct ibv_send_wr** badWr, struct WrExpRsp* expRsp);
} // namespace hccl
#endif // end HCCL_ADAPTER_VERBS_H
