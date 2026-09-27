/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef USER_REMOTE_MEM_GETTER_H
#define USER_REMOTE_MEM_GETTER_H

#include <cstdint>
#include <vector>
#include <array>
#include <string>
#include <functional>
#include "hccl_mem_defs.h"
#include "hccl/hccl_types.h"
#include "log.h"

constexpr uint32_t MAX_BUFFER_NUM = 30000;

namespace Hccl {

template <typename T>
struct RemoteMemCtx {
    bool& cacheValid;
    std::vector<T>& rmtBufferVec;
    std::vector<CommMem>& remoteUserMems;
    std::vector<std::string>& memInfoCopies;
    std::vector<char*>& memInfoPointers;
    CommMem** remoteMem;
    char*** memInfos;
    uint32_t* memNum;

    RemoteMemCtx(
        bool& cacheValid, std::vector<T>& rmtBufferVec, std::vector<CommMem>& remoteUserMems,
        std::vector<std::string>& memInfoCopies, std::vector<char*>& memInfoPointers, CommMem** remoteMem,
        char*** memInfos, uint32_t* memNum)
        : cacheValid(cacheValid),
          rmtBufferVec(rmtBufferVec),
          remoteUserMems(remoteUserMems),
          memInfoCopies(memInfoCopies),
          memInfoPointers(memInfoPointers),
          remoteMem(remoteMem),
          memInfos(memInfos),
          memNum(memNum) {};
};

inline HcclMemType CommMemTypeToHcclMemType(CommMemType type)
{
    switch (type) {
        case COMM_MEM_TYPE_DEVICE:
            return HCCL_MEM_TYPE_DEVICE;
        case COMM_MEM_TYPE_HOST:
            return HCCL_MEM_TYPE_HOST;
        default:
            return HCCL_MEM_TYPE_NUM;
    }
}

inline CommMemType HcclMemTypeToCommMemType(HcclMemType type)
{
    switch (type) {
        case HCCL_MEM_TYPE_DEVICE:
            return COMM_MEM_TYPE_DEVICE;
        case HCCL_MEM_TYPE_HOST:
            return COMM_MEM_TYPE_HOST;
        default:
            return COMM_MEM_TYPE_INVALID;
    }
}

template <typename T>
HcclResult GetRemoteUserMems(RemoteMemCtx<T>& remoteMemCtx)
{
    CHK_PRT_RET(!remoteMemCtx.remoteMem, HCCL_ERROR("[GetRemoteUserMems] remoteMem is nullptr"), HCCL_E_PARA);
    CHK_PRT_RET(!remoteMemCtx.memInfos, HCCL_ERROR("[GetRemoteUserMems] memInfos is nullptr"), HCCL_E_PARA);
    CHK_PRT_RET(!remoteMemCtx.memNum, HCCL_ERROR("[GetRemoteUserMems] memNum is nullptr"), HCCL_E_PARA);
    *(remoteMemCtx.remoteMem) = nullptr;
    *(remoteMemCtx.memInfos) = nullptr;
    *(remoteMemCtx.memNum) = 0;
    uint32_t userMemCount = remoteMemCtx.rmtBufferVec.size();
    if (userMemCount == 0) {
        HCCL_WARNING("[%s] bufferNum is 0.", __func__);
        return HCCL_SUCCESS;
    }
    // 检查是否有缓存
    if (!remoteMemCtx.cacheValid) {
        remoteMemCtx.remoteUserMems.clear();
        remoteMemCtx.memInfoCopies.clear();
        remoteMemCtx.memInfoCopies.reserve(userMemCount);
        remoteMemCtx.memInfoPointers.clear();
        remoteMemCtx.memInfoPointers.reserve(userMemCount);
        for (uint32_t i = 0; i < userMemCount; ++i) {
            auto& rmtBuffer = remoteMemCtx.rmtBufferVec[i];
            if (rmtBuffer == nullptr) {
                return HCCL_E_PTR;
            }
            CommMem mem{};
            mem.type = HcclMemTypeToCommMemType(rmtBuffer->GetMemType());
            mem.addr = reinterpret_cast<void*>(rmtBuffer->GetAddr());
            mem.size = rmtBuffer->GetSize();
            remoteMemCtx.remoteUserMems.push_back(mem);
            std::string memInfoCopy = rmtBuffer->GetMemInfo();
            remoteMemCtx.memInfoCopies.push_back(std::move(memInfoCopy));
            remoteMemCtx.memInfoPointers.push_back(&remoteMemCtx.memInfoCopies.back()[0]);
            HCCL_INFO(
                "[%s] Found buffer[addr:%p, size:%llu, memInfo:%s]", __func__, mem.addr, mem.size,
                remoteMemCtx.memInfoCopies.back().c_str());
        }
        remoteMemCtx.cacheValid = true;
    }
    *(remoteMemCtx.remoteMem) = remoteMemCtx.remoteUserMems.data();
    *(remoteMemCtx.memInfos) = remoteMemCtx.memInfoPointers.data();
    *(remoteMemCtx.memNum) = userMemCount;
    return HCCL_SUCCESS;
}
} // namespace Hccl

#endif // USER_REMOTE_MEM_GETTER_H
