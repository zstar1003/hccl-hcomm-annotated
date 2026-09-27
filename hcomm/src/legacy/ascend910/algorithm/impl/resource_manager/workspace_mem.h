/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef WORKSPACE_MEM_H
#define WORKSPACE_MEM_H

#include <string>
#include <cstdio>
#include <mutex>
#include <map>
#include "hccl/base.h"

namespace hccl {
using WorkSpaceMem_t = struct tagWorkSpaceMem {
    void* curPtr{nullptr};
    u64 maxSize{0};   /* 最大的内存大小 */
    u64 totalSize{0}; /* 当前已累积的内存大小 */
};

class WorkSpaceMem {
public:
    WorkSpaceMem();

    ~WorkSpaceMem();

    /* 设置全局资源 */
    HcclResult SetMemResource(const std::string& tag, void* ptr, u64 maxSize);

    /* 内存分配 */
    void* AllocMem(const std::string& tag, u64 size);

    /* 基于tag销毁资源  */
    HcclResult DestroyMemResource(const std::string& tag);

    /* 销毁全局资源 */
    void DestroyMemResource();

    bool IsExist(const std::string& tag);

    std::map<std::string, WorkSpaceMem_t> memResMap_;

private:
    std::mutex memResMutex_;
};
} // namespace hccl
#endif /* * WORKSPACE_MEM_H */
