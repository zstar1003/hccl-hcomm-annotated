/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log.h"
#include "comm_topo_desc.h"

namespace hccl {
CommTopoDesc& CommTopoDesc::GetInstance()
{
    static CommTopoDesc commTopoDesc;
    return commTopoDesc;
}

CommTopoDesc::CommTopoDesc() {}

CommTopoDesc::~CommTopoDesc() {}

void CommTopoDesc::SaveRankSize(std::string& str, uint32_t rankSize)
{
    std::unique_lock<std::mutex> topoDescLock(lock_);
    rankSizeMap_[str] = rankSize;
}

void CommTopoDesc::SaveL0TopoType(std::string& str, CommTopo topoType)
{
    std::unique_lock<std::mutex> topoDescLock(lock_);
    l0TopoTypeMap_[str] = topoType;
}

HcclResult CommTopoDesc::GetRankSize(std::string& str, uint32_t* rankSize)
{
    std::unique_lock<std::mutex> topoDescLock(lock_);
    auto it = rankSizeMap_.find(str);
    if (it != rankSizeMap_.end()) {
        *rankSize = it->second;
        return HCCL_SUCCESS;
    } else {
        HCCL_ERROR("commName[%s] not found, please check comm init", str.c_str());
        return HCCL_E_PARA;
    }
}

HcclResult CommTopoDesc::GetL0TopoType(std::string& str, CommTopo* topoType)
{
    std::unique_lock<std::mutex> topoDescLock(lock_);
    auto it = l0TopoTypeMap_.find(str);
    if (it != l0TopoTypeMap_.end()) {
        *topoType = it->second;
        return HCCL_SUCCESS;
    } else {
        HCCL_ERROR("commName[%s] not found, please check comm init", str.c_str());
        return HCCL_E_PARA;
    }
}
} // namespace hccl
