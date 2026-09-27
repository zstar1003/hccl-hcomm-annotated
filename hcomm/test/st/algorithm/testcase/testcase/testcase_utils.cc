/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "testcase_utils.h"
#include <securec.h>
#include <stdlib.h>
#include "checker.h"
using namespace checker;

#include "sub_inc/mmpa_typedef_linux.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

int memcpy_s(void* dest, size_t destMax, const void* src, size_t count)
{
    memcpy(dest, src, count);
    return 0;
}

INT32 mmGetEnv(const CHAR* name, CHAR* value, UINT32 len)
{
    INT32 ret;
    UINT32 envLen = 0;
    if ((name == NULL) || (value == NULL) || (len == MMPA_ZERO)) {
        return EN_INVALID_PARAM;
    }
    const CHAR* envPtr = getenv(name);
    if (envPtr == NULL) {
        return EN_ERROR;
    }

    UINT32 lenOfRet = (UINT32)strlen(envPtr);
    if (lenOfRet < (UINT32)(MMPA_MEM_MAX_LEN - 1)) {
        envLen = lenOfRet + 1U;
    }

    if ((envLen != MMPA_ZERO) && (len < envLen)) {
        return EN_INVALID_PARAM;
    } else {
        ret = memcpy_s(value, len, envPtr, envLen); // lint !e613
        if (ret != EN_OK) {
            return EN_ERROR;
        }
    }
    return EN_OK;
}

#ifdef __cplusplus
}
#endif // __cplusplus

void ClearHcclEnv()
{
    unsetenv("HCCL_HIGH_PERF_ENABLE");
    unsetenv("HCCL_DETERMINISTIC");
    unsetenv("HCCL_INTRA_PCIE_ENABLE");
    unsetenv("HCCL_INTRA_ROCE_ENABLE");
    unsetenv("HCCL_ALGO");
    unsetenv("HCCL_BUFFSIZE");
    unsetenv("HCCL_INTER_HCCS_DISABLE");
    unsetenv("HCCL_OP_EXPANSION_MODE");
    unsetenv("HCCL_CONCURRENT_ENABLE");
    unsetenv("HCCL_DEBUG_CONFIG");

    CheckerReset();
    return;
}

std::vector<u64> GenerateSendCountMatrix(u64 count, u32 rankSize)
{
    std::vector<u64> sendCountMatrix(rankSize * rankSize, count);
    return sendCountMatrix;
}

void GenAllToAllVParams(
    u32 rankSize, u64 count, std::vector<u64>& sendCounts, std::vector<u64>& sdispls, std::vector<u64>& recvCounts,
    std::vector<u64>& rdispls)
{
    u64 sendDisplacement = 0;
    u64 recvDisplacement = 0;
    for (u32 i = 0; i < rankSize; i++) {
        sendCounts.push_back(count);
        sdispls.push_back(sendDisplacement);
        recvCounts.push_back(count);
        rdispls.push_back(recvDisplacement);
        sendDisplacement += count;
        recvDisplacement += count;
    }
    return;
}
