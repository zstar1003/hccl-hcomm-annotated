/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <vector>
#include <memory>
#include "hccl/hccl_res.h"
#include "hccl_comm_pub.h"
#include "coll_comm_mgr.h"
#include "orion_adapter_rts.h"
#include "hcom_common.h"
#include "param_check_basic_v2.h"
#include "log.h"

#ifdef __cplusplus
extern "C" {
#endif

HcclResult HcomSetAttachedStream(const char* group, u32 graphId, const rtStream_t* stream, s32 len)
{
    EXCEPTION_HANDLE_BEGIN
    HCCL_INFO(
        "[HcomSetAttachedStream] entry, group[%s], graphId[%u], stream[%p], len[%d]",
        group == nullptr ? "nullptr" : group, graphId, stream, len);

    CHK_PRT_RET(len < 0, HCCL_ERROR("[HcomSetAttachedStream] len is %d", len), HCCL_E_PARA);
    CHK_PTR_NULL(stream);
    if (group == nullptr) {
        group = HCCL_WORLD_GROUP;
    }

    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        return HCCL_SUCCESS;
        if (len == 0) {
            HCCL_WARNING("[HcomSetAttachedStream] len is 0, no stream");
            return HCCL_SUCCESS;
        }
        void* attachedStream = stream[0];
        s32 deviceLogicId = Hccl::HrtGetDevice();
        auto& mgr = hccl::CollCommMgr::GetInstance().GetOrderLaunchThreadMgr(deviceLogicId);
        return mgr.SetAttachedStream(std::string(group), graphId, attachedStream);
    }());

    std::shared_ptr<hccl::hcclComm> hcclComm = nullptr;
    std::vector<rtStream_t> rtStream(stream, stream + len);
    if (HcomGetCommByGroup(group, hcclComm) == HCCL_SUCCESS) {
        CHK_RET(hcclComm->SetAttachedStream(graphId, rtStream));
    } else {
        HCCL_WARNING("[HcomSetAttachedStream] HcclCommBase now don't support set attached stream");
    }
    EXCEPTION_HANDLE_END
    return HCCL_SUCCESS;
}

#ifdef __cplusplus
}
#endif
