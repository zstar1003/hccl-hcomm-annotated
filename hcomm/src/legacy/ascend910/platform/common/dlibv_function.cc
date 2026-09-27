/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dlibv_function.h"

#include <string>
#include <map>
#include "hccl_dl.h"
#include "log.h"

namespace hccl {
DlIbvFunction& DlIbvFunction::GetInstance()
{
    static DlIbvFunction hcclDlIbvFunction;
    return hcclDlIbvFunction;
}

DlIbvFunction::DlIbvFunction() : handle_(nullptr) {}

DlIbvFunction::~DlIbvFunction()
{
    if (handle_ != nullptr) {
        (void)HcclDlclose(handle_);
        handle_ = nullptr;
    }
}

HcclResult DlIbvFunction::DlIbvFunctionApiInit()
{
    dlRcoeGetCqEvent = (s32(*)(struct ibv_comp_channel * channel, struct ibv_cq * *cq, void** cq_context))
        HcclDlsym(handle_, "ibv_get_cq_event");
    CHK_SMART_PTR_NULL(dlRcoeGetCqEvent);

    dlRcoeAckCqEvent = (void (*)(struct ibv_cq* qp, unsigned int nevents))HcclDlsym(handle_, "ibv_ack_cq_events");
    CHK_SMART_PTR_NULL(dlRcoeAckCqEvent);

    dlRcoeQueryQp
        = (s32(*)(struct ibv_qp * qp, struct ibv_qp_attr * attr, int attr_mask, struct ibv_qp_init_attr* init_attr))
            HcclDlsym(handle_, "ibv_query_qp");
    CHK_SMART_PTR_NULL(dlRcoeQueryQp);

    return HCCL_SUCCESS;
}

HcclResult DlIbvFunction::DlIbvFunctionInit()
{
    std::lock_guard<std::mutex> lock(handleMutex_);
    if (handle_ == nullptr) {
        handle_ = HcclDlopen("libibverbs.so", RTLD_NOW);
        const char* errMsg = dlerror();
        CHK_PRT_RET(
            handle_ == nullptr,
            HCCL_ERROR(
                "dlopen [%s] failed, %s", "libibverbs.so",
                (errMsg == nullptr) ? "please check the file exist or permission denied." : errMsg),
            HCCL_E_OPEN_FILE_FAILURE);
    }

    CHK_RET(DlIbvFunctionApiInit());
    return HCCL_SUCCESS;
}
} // namespace hccl
