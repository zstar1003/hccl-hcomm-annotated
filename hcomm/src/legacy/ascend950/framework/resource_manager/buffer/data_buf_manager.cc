/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "data_buf_manager.h"
#include "log.h"
#include "exception_util.h"
namespace Hccl {

DataBufManager::~DataBufManager() { DECTOR_TRY_CATCH("DataBufManager", Destroy()); }

bool DataBufManager::IsExist(const std::string& opTag, BufferType type)
{
    return bufs.find(opTag) != bufs.end() && bufs[opTag].find(type) != bufs[opTag].end();
}

Buffer* DataBufManager::Register(const std::string& opTag, BufferType type, std::shared_ptr<Buffer> buffer)
{
    if (buffer == nullptr) {
        HCCL_WARNING("opTag[%s] type[%s] 's buffer is nullptr", opTag.c_str(), type.Describe().c_str());
        return nullptr;
    }
    bufs[opTag][type] = buffer;
    return bufs[opTag][type].get();
}

void DataBufManager::Deregister(const std::string& opTag, BufferType type)
{
    if (!IsExist(opTag, type)) {
        HCCL_WARNING("Cannot find Buffer in map.");
        return;
    }
    bufs[opTag].erase(type);
    if (bufs[opTag].empty()) {
        bufs.erase(opTag);
    }
}

HcclResult DataBufManager::Deregister(const std::string& opTag)
{
    if (bufs.find(opTag) == bufs.end()) {
        HCCL_WARNING("[DataBufManager::%s] opTag[%s] Cannot find Buffer in bufs.", __func__, opTag.c_str());
        return HCCL_SUCCESS;
    }
    bufs.erase(opTag);
    return HCCL_SUCCESS;
}

Buffer* DataBufManager::Get(const std::string& opTag, BufferType type)
{
    if (bufs.find(opTag) != bufs.end() && bufs[opTag].find(type) != bufs[opTag].end()) {
        return bufs[opTag][type].get();
    }
    HCCL_WARNING("Buffer doesn't exist, opTag[%s], dataBuf[type=%s]", opTag.c_str(), type.Describe().c_str());
    return nullptr;
}

void DataBufManager::Destroy() { bufs.clear(); }

} // namespace Hccl
