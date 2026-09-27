/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "adapter_verbs.h"
#include <functional>

#include "log.h"
#include "dlibv_function.h"

namespace hccl {
HcclResult hrtIbvPostSrqRecv(struct ibv_srq* srq, struct ibv_recv_wr* wr, struct ibv_recv_wr** badWr)
{
    s32 ret = ibv_post_srq_recv(srq, wr, badWr);
    CHK_PRT_RET(ret != 0, HCCL_ERROR("ibv_post_srq_recv failed. errno:%d ret: %d", errno, ret), HCCL_E_NETWORK);
    return HCCL_SUCCESS;
}

HcclResult hrtIbvPostRecv(struct ibv_qp* qp, struct ibv_recv_wr* wr, struct ibv_recv_wr** badWr)
{
    s32 ret = ibv_post_recv(qp, wr, badWr);
    CHK_PRT_RET(ret == ENOMEM, HCCL_WARNING("post recv wqe overflow.[%d]", ret), HCCL_E_AGAIN);
    CHK_PRT_RET(ret != 0, HCCL_ERROR("ibv_post_recv failed. errno:%d ret: %d", errno, ret), HCCL_E_NETWORK);
    return HCCL_SUCCESS;
}

HcclResult hrtIbvPostSend(struct ibv_qp* qp, struct ibv_send_wr* wr, struct ibv_send_wr** badWr)
{
    s32 ret = ibv_post_send(qp, wr, badWr);
    CHK_PRT_RET(ret == ENOMEM, HCCL_WARNING("post send wqe overflow.[%d]", ret), HCCL_E_AGAIN);
    CHK_PRT_RET(ret != 0, HCCL_ERROR("ibv_post_send failed. errno:%d", ret), HCCL_E_NETWORK);
    return HCCL_SUCCESS;
}

HcclResult hrtIbvPollCq(struct ibv_cq* cq, int maxNum, struct ibv_wc* wc, s32& num)
{
    int pollNum = 0;
    pollNum = ibv_poll_cq(cq, maxNum, wc);
    num = pollNum;
    CHK_PRT_RET((num < 0), HCCL_ERROR("ibv_poll_cq failed. num:%d", num), HCCL_E_NETWORK);
    return HCCL_SUCCESS;
}

HcclResult hrtIbvReqNotifyCq(struct ibv_cq* cq, int solicitedOnly)
{
    s32 ret = ibv_req_notify_cq(cq, solicitedOnly);
    CHK_PRT_RET(ret != 0, HCCL_ERROR("ibv_req_notify_cq failed. errno:%d", ret), HCCL_E_NETWORK);
    return HCCL_SUCCESS;
}

HcclResult hrtIbvGetCqEvent(struct ibv_comp_channel* channel, struct ibv_cq** cq, void** cq_context)
{
    s32 ret = DlIbvFunction::GetInstance().dlRcoeGetCqEvent(channel, cq, cq_context);
    CHK_PRT_RET(ret != 0, HCCL_ERROR("hrtIbvGetCqEvent failed. errno:%d", ret), HCCL_E_NETWORK);
    return HCCL_SUCCESS;
}

void hrtIbvAckCqEvent(struct ibv_cq* qp, unsigned int nevents)
{
    DlIbvFunction::GetInstance().dlRcoeAckCqEvent(qp, nevents);
    return;
}

HcclResult hrtIbvQueryQp(struct ibv_qp* qp)
{
    struct ibv_qp_attr attr {};
    struct ibv_qp_init_attr init_attr {};
    CHK_RET(DlIbvFunction::GetInstance().DlIbvFunctionInit());
    s32 ret = DlIbvFunction::GetInstance().dlRcoeQueryQp(qp, &attr, IBV_QP_STATE, &init_attr);
    CHK_PRT_RET(ret != 0, HCCL_ERROR("hrtIbvQueryQp failed. errno:%d", ret), HCCL_E_NETWORK);
    return HCCL_SUCCESS;
}

HcclResult
HrtHnsIbvExpPostSend(struct ibv_qp* qp, struct ibv_send_wr* wr, struct ibv_send_wr** badWr, struct WrExpRsp* expRsp)
{
    CHK_PTR_NULL(qp);
    CHK_PTR_NULL(wr);
    CHK_PTR_NULL(badWr);
    CHK_PTR_NULL(expRsp);
    static bool init = false;
    if (UNLIKELY(!init)) {
        CHK_RET(DlHnsFunction::GetInstance().DlHnsFunctionInit());
        init = true;
    }

    s32 ret = DlHnsFunction::GetInstance().dlHnsIbvExpPostSend(qp, wr, badWr, expRsp);
    CHK_PRT_RET(
        ret == -ENOENT || ret == -EAGAIN || ret == -ENOMEM || ret == ENOENT || ret == EAGAIN || ret == ENOMEM,
        HCCL_WARNING("HrtHnsIbvExpPostSend failed. errno:%d", ret), HCCL_E_AGAIN);
    CHK_PRT_RET(ret != 0, HCCL_ERROR("HrtHnsIbvExpPostSend failed. errno:%d", ret), HCCL_E_NETWORK);
    return HCCL_SUCCESS;
}
} // namespace hccl
