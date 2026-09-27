/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hcomm_mem_alloc.h"

#include "acl/acl_rt.h"
#include "log.h"
#include "param_check_pub.h"

namespace hcomm {
namespace {
#define ALIGN_SIZE(size, align) ({ (size) = (((size) + (align) - 1) / (align)) * (align); })
} // namespace

HcommResult MemAlloc(void** ptr, size_t size)
{
    CHK_PTR_NULL(ptr);
    CHK_PRT_RET(size == 0, HCCL_ERROR("[%s] size is zero", __func__), HCCL_E_PARA);

    aclError ret = ACL_SUCCESS;
    int32_t deviceId = 0;
    ret = aclrtGetDevice(&deviceId);
    CHK_PRT_RET(ret != ACL_SUCCESS, HCCL_ERROR("[%s] GetDevice failed, ret[%d]", __func__, ret), HCCL_E_RUNTIME);

    aclrtPhysicalMemProp prop;
    prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
    prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
    prop.memAttr = ACL_HBM_MEM_HUGE;
    prop.location.id = deviceId;
    prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
    prop.reserve = 0;

    size_t allocSize = size;
    size_t granularity = 0;
    ret = aclrtMemGetAllocationGranularity(&prop, ACL_RT_MEM_ALLOC_GRANULARITY_RECOMMENDED, &granularity);
    CHK_PRT_RET(
        ret != ACL_SUCCESS || granularity == 0,
        HCCL_ERROR("[%s] GetAllocationGranularity failed, granularity[%llu], ret[%d]", __func__, granularity, ret),
        HCCL_E_RUNTIME);
    ALIGN_SIZE(allocSize, granularity);
    HCCL_INFO(
        "[%s] deviceId[%d], granularity[%llu], size[%llu], allocSize[%llu].", __func__, deviceId, granularity, size,
        allocSize);

    ret = aclrtReserveMemAddress(ptr, allocSize, 0, nullptr, 1);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR("[%s] ReserveMemAddress failed, virPtr[%p] size[%llu], ret[%d]", __func__, ptr, allocSize, ret),
        HCCL_E_RUNTIME);

    void* virPtr = *ptr;
    aclrtDrvMemHandle handle;
    ret = aclrtMallocPhysical(&handle, allocSize, &prop, 0);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR("[%s] MallocPhysical failed, size[%llu], ret[%d]", __func__, allocSize, ret);
        aclrtReleaseMemAddress(virPtr);
        return HCCL_E_RUNTIME;
    }
    HCCL_INFO("[%s] Start to MapMem virPtr[%p], handle[%p]", __func__, virPtr, handle);
    ret = aclrtMapMem(virPtr, allocSize, 0, handle, 0);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR(
            "[%s] MapMem virPtr[%p] size[%llu] handle[%p] failed, ret[%d]", __func__, virPtr, allocSize, handle, ret);
        aclrtFreePhysical(handle);
        aclrtReleaseMemAddress(virPtr);
        return HCCL_E_RUNTIME;
    }

    return HCCL_SUCCESS;
}

HcommResult MemFree(void* ptr)
{
    if (ptr == nullptr) {
        HCCL_DEBUG("[%s] virPtr is nullptr.", __func__);
        return HCCL_SUCCESS;
    }
    aclError ret = ACL_SUCCESS;
    aclrtDrvMemHandle handle;
    ret = aclrtMemRetainAllocationHandle(ptr, &handle);
    CHK_PRT_RET(
        ret != ACL_SUCCESS, HCCL_ERROR("[%s] RetainAllocationHandle virPtr[%p] failed, ret[%d]", __func__, ptr, ret),
        HCCL_E_RUNTIME);
    HCCL_INFO("[%s] Start to UnmapMem virPtr[%p], handle[%p]", __func__, ptr, handle);
    ret = aclrtUnmapMem(ptr);
    CHK_PRT_RET(
        ret != ACL_SUCCESS, HCCL_ERROR("[%s] UnmapMem virPtr[%p] failed, ret[%d]", __func__, ptr, ret), HCCL_E_RUNTIME);
    ret = aclrtFreePhysical(handle);
    CHK_PRT_RET(
        ret != ACL_SUCCESS, HCCL_ERROR("[%s] FreePhysical handle[%p] failed, ret[%d]", __func__, handle, ret),
        HCCL_E_RUNTIME);
    ret = aclrtReleaseMemAddress(ptr);
    CHK_PRT_RET(
        ret != ACL_SUCCESS, HCCL_ERROR("[%s] ReleaseMemAddress virPtr[%p] failed, ret[%d]", __func__, ptr, ret),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
}
} // namespace hcomm
