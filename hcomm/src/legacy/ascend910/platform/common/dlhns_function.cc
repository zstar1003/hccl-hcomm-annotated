/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dlhns_function.h"

#include <string>
#include <map>
#include "hccl_dl.h"
#include "log.h"

namespace hccl {

DlHnsFunction& DlHnsFunction::GetInstance()
{
    static DlHnsFunction hcclDlHnsFunction;
    return hcclDlHnsFunction;
}

DlHnsFunction::DlHnsFunction() : handle_(nullptr) {}

DlHnsFunction::~DlHnsFunction()
{
    if (handle_ != nullptr) {
        (void)HcclDlclose(handle_);
        handle_ = nullptr;
    }
}

HcclResult DlHnsFunction::DlHnsFunctionRoceInit()
{
    dlHnsIbvExtPostSend = (int (*)(
        struct ibv_qp*, struct ibv_send_wr*, struct ibv_send_wr**, struct IbvPostSendExtAddt*,
        struct IbvPostSendExtResp*))HcclDlsym(handle_, "ibv_ext_post_send");
    CHK_SMART_PTR_NULL(dlHnsIbvExtPostSend);

    dlHnsIbvExpPostSend
        = (int (*)(struct ibv_qp*, struct ibv_send_wr*, struct ibv_send_wr**, struct WrExpRsp*))HcclDlsym(
            handle_, "ibv_exp_post_send");
    CHK_SMART_PTR_NULL(dlHnsIbvExpPostSend);
    return HCCL_SUCCESS;
}

HcclResult DlHnsFunction::DlHnsFunctionSoInit()
{
    if (handle_ == nullptr) {
        handle_ = HcclDlopen("libhns-rdmav25.so", RTLD_NOW);
        if (handle_ != nullptr) {
            return HCCL_SUCCESS;
        }
        return HCCL_E_INTERNAL;
    } else {
        HCCL_INFO("roce_user_api dlopen again!");
    }
    return HCCL_SUCCESS;
}

HcclResult DlHnsFunction::DlHnsFunctionInit()
{
    std::lock_guard<std::mutex> lock(handleMutex_);
    CHK_RET(DlHnsFunctionSoInit());
    CHK_RET(DlHnsFunctionRoceInit());

    return HCCL_SUCCESS;
}

} // namespace hccl
