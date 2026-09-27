/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cast_utils.h"
#include "hcomm_c_adpt.h"
#include "hcomm_res_defs.h"
#include "log.h"
#include "param_check_pub.h"
#include "hcom_common.h"
#include "comm_engine_utils.h"

HcommResult HcommEngineCtxCreate(CommEngine engine, uint64_t size, void** ctx)
{
    CHK_PTR_NULL(ctx);
    if (engine == COMM_ENGINE_CPU || engine == COMM_ENGINE_CPU_TS || engine == COMM_ENGINE_CCU) {
        *ctx = malloc(size);
        CHK_PTR_NULL(*ctx);
        auto ret = memset_s(*ctx, size, 0, size);
        if (ret != EOK) {
            HCCL_ERROR("[%s] memset_s failed, ret[%d]", __func__, ret);
            free(*ctx);
            *ctx = nullptr;
            return HCCL_E_INTERNAL;
        }
    } else if (engine == COMM_ENGINE_AICPU || engine == COMM_ENGINE_AICPU_TS || engine == COMM_ENGINE_AIV) {
        CHK_RET(hrtMalloc(ctx, size));
    } else {
        HCCL_ERROR(
            "[%s] not support engine type[%s]", __func__, GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str());
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcommResult HcommEngineCtxDestroy(CommEngine engine, void* ctx)
{
    CHK_PTR_NULL(ctx);
    if (engine == COMM_ENGINE_CPU || engine == COMM_ENGINE_CPU_TS || engine == COMM_ENGINE_CCU) {
        free(ctx);
    } else if (engine == COMM_ENGINE_AICPU || engine == COMM_ENGINE_AICPU_TS || engine == COMM_ENGINE_AIV) {
        CHK_RET(hrtFree(ctx));
    } else {
        HCCL_ERROR("[%s] invalid engine[%s]", __func__, GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str());
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcommResult HcommEngineCtxCopy(CommEngine engine, void* dstCtx, const void* srcCtx, uint64_t size)
{
    CHK_PTR_NULL(dstCtx);
    CHK_PTR_NULL(srcCtx);
    if (engine == COMM_ENGINE_AICPU_TS || engine == COMM_ENGINE_AICPU || engine == COMM_ENGINE_AIV) {
        // 从Host内存拷贝到Device Context内存上
        CHK_RET(hrtMemSyncCopy(
            ReinterpretAs<uint8_t*>(dstCtx), size, srcCtx, size, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    } else if (engine == COMM_ENGINE_CPU || engine == COMM_ENGINE_CPU_TS || engine == COMM_ENGINE_CCU) {
        CHK_SAFETY_FUNC_RET(memcpy_s(ReinterpretAs<uint8_t*>(dstCtx), size, srcCtx, size));
    } else {
        HCCL_ERROR(
            "[%s] copy engine ctx failed, Unsupported engine[%s]", __func__,
            GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str());
        return HCCL_E_PARA;
    }
    HCCL_INFO(
        "[%s]copy engine ctx success, engine[%s]", __func__,
        GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str());
    return HCCL_SUCCESS;
}
