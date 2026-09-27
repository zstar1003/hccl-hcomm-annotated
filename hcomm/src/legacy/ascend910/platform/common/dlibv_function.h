/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_SRC_DLIBVFUNCTION_H
#define HCCL_SRC_DLIBVFUNCTION_H

#include <functional>
#include <mutex>
#include <hccl/hccl_types.h>

#include "hccl/base.h"
#include "ascend_hal.h"

namespace hccl {
class DlIbvFunction {
public:
    virtual ~DlIbvFunction();
    static DlIbvFunction& GetInstance();
    HcclResult DlIbvFunctionInit();
    std::function<s32(struct ibv_comp_channel* channel, struct ibv_cq** cq, void** cq_context)> dlRcoeGetCqEvent
        = nullptr;
    std::function<void(struct ibv_cq* qp, unsigned int nevents)> dlRcoeAckCqEvent = nullptr;
    std::function<s32(struct ibv_qp* qp, struct ibv_qp_attr* attr, int attr_mask, struct ibv_qp_init_attr* init_attr)>
        dlRcoeQueryQp = nullptr;

protected:
private:
    void* handle_;
    std::mutex handleMutex_;
    DlIbvFunction(const DlIbvFunction&);
    DlIbvFunction& operator=(const DlIbvFunction&);
    DlIbvFunction();
    HcclResult DlIbvFunctionApiInit();
};
} // namespace hccl

#endif // HCCL_SRC_DLIBVFUNCTION_H
